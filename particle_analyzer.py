#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Analysis Module
Detects particles in images and generates pixel distribution maps based on scale bar detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
from typing import List, Tuple, Dict, Optional
from lightweight_scale_detector import LightweightScaleDetectorWithPreprocessing
from scale_visualizer import ScaleVisualizer
import seaborn as sns
from scipy import ndimage
from skimage import measure, morphology
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import pandas as pd
from mock_ocr import process as ocr_process
import os

class ParticleAnalyzer:
    """Particle Analyzer"""
    
    def __init__(self, scale_detector: LightweightScaleDetectorWithPreprocessing):
        self.scale_detector = scale_detector
        self.scale_length_pixels = None  # Scale bar length in pixels
        self.scale_length_real = None    # Actual scale bar length (units: micrometers, etc.)
        self.pixel_to_real_ratio = None  # Conversion ratio from pixels to real units
        
    def set_scale_reference(self, scale_length_real: float, unit: str = "μm"):
        """
        Set scale bar reference
        Args:
            scale_length_real: Actual length of the scale bar
            unit: Unit (micrometers, millimeters, etc.)
        """
        self.scale_length_real = scale_length_real
        self.unit = unit
        
    def detect_particles(self, 
                        image: np.ndarray, 
                        method: str = "watershed",
                        min_area: int = 50,
                        max_area: int = 5000,
                        circularity_threshold: float = 0.3,
                        ocr_boxes: list = None) -> Dict:
        """
        Detect particles in the image
        Args:
            image: Input image
            method: Detection method ("watershed", "contour", "blob")
            min_area: Minimum particle area
            max_area: Maximum particle area
            circularity_threshold: Circularity threshold
        Returns:
            Dictionary containing particle information
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Preprocessing
        blurred = gaussian(gray, sigma=1)
        
        if method == "watershed":
            particles = self._detect_particles_watershed(blurred, min_area, max_area, ocr_boxes)
        elif method == "contour":
            particles = self._detect_particles_contour(blurred, min_area, max_area, circularity_threshold)
        elif method == "blob":
            particles = self._detect_particles_blob(blurred, min_area, max_area)
        else:
            raise ValueError(f"Unsupported detection method: {method}")
        
        return particles
    
    def _detect_particles_watershed(self, 
                                  image: np.ndarray, 
                                  min_area: int, 
                                  max_area: int,
                                  ocr_boxes: list = None) -> Dict:
        """Detect particles using watershed algorithm"""
        # Binarization - try multiple methods
        try:
            thresh = threshold_otsu(image)
            binary = image < thresh
        except:
            # If Otsu fails, use fixed threshold
            binary = image < 128
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = morphology.binary_opening(binary, kernel)
        binary = morphology.binary_closing(binary, kernel)
        
        # Distance transform
        distance = ndimage.distance_transform_edt(binary)
        
        # Find local maxima
        local_max = peak_local_max(distance, min_distance=10, labels=binary)
        local_max_mask = np.zeros_like(binary, dtype=bool)
        local_max_mask[tuple(local_max.T)] = True
        
        # Markers
        markers = measure.label(local_max_mask)
        
        # Watershed segmentation
        labels = watershed(-distance, markers, mask=binary)
        
        # Analyze regions
        regions = measure.regionprops(labels)
        
        particles = {
            'centroids': [],
            'areas': [],
            'diameters': [],
            'circularities': [],
            'bboxes': [],
            'labels': []
        }
        
        # Filter particles by excluding contours overlapping with ocr_boxes
        filtered = 0
        for region in regions:
            if min_area <= region.area <= max_area:
                # Calculate bounding box
                minr, minc, maxr, maxc = region.bbox
                region_box = (minc, minr, maxc, maxr)
                overlap = False
                if ocr_boxes:
                    for ocr_box in ocr_boxes:
                        # Debug output
                        print(f"[Debug] region_box: {region_box}, ocr_box: {ocr_box}")
                        # Check for overlap
                        if not (region_box[2] < ocr_box[0] or ocr_box[2] < region_box[0] or region_box[3] < ocr_box[1] or ocr_box[3] < region_box[1]):
                            overlap = True
                            print(f"[Debug] overlap=True, this contour is excluded")
                            break
                if not overlap:
                    # Only keep particles not overlapping with OCR boxes
                    particles['centroids'].append(region.centroid)
                    particles['areas'].append(region.area)
                    particles['diameters'].append(region.equivalent_diameter)
                    particles['circularities'].append(region.extent)
                    particles['bboxes'].append(region.bbox)
                    particles['labels'].append(region.label)
                else:
                    filtered += 1
        if ocr_boxes:
            print(f"[Filter] {filtered} contours were excluded due to overlap with OCR boxes")
        
        return particles
    
    def _detect_particles_contour(self, 
                                image: np.ndarray, 
                                min_area: int, 
                                max_area: int,
                                circularity_threshold: float) -> Dict:
        """Detect particles using contour detection"""
        # Binarization - try multiple methods
        try:
            thresh = threshold_otsu(image)
            binary = image < thresh
        except:
            # If Otsu fails, use fixed threshold
            binary = image < 128
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = morphology.binary_opening(binary, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        particles = {
            'centroids': [],
            'areas': [],
            'diameters': [],
            'circularities': [],
            'bboxes': [],
            'contours': []
        }
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                if circularity >= circularity_threshold:
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    
                    # Calculate equivalent diameter
                    diameter = np.sqrt(4 * area / np.pi)
                    
                    # Bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    particles['centroids'].append((cy, cx))  # OpenCV uses (y, x)
                    particles['areas'].append(area)
                    particles['diameters'].append(diameter)
                    particles['circularities'].append(circularity)
                    particles['bboxes'].append((y, x, y+h, x+w))
                    particles['contours'].append(contour)
        
        return particles
    
    def _detect_particles_blob(self, 
                             image: np.ndarray, 
                             min_area: int, 
                             max_area: int) -> Dict:
        """Detect particles using blob detection"""
        # Ensure image is 8-bit
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Use SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByArea = True
        params.filterByCircularity = True
        params.minCircularity = 0.3
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)
        
        particles = {
            'centroids': [],
            'areas': [],
            'diameters': [],
            'circularities': [],
            'bboxes': [],
            'keypoints': keypoints
        }
        
        for kp in keypoints:
            x, y = kp.pt
            size = kp.size
            
            particles['centroids'].append((y, x))
            particles['areas'].append(np.pi * (size/2)**2)
            particles['diameters'].append(size)
            particles['circularities'].append(1.0)  # Assume circular shape
            particles['bboxes'].append((y-size/2, x-size/2, y+size/2, x+size/2))
        
        return particles
    
    def analyze_particle_distribution(self, particles: Dict) -> Dict:
        """Analyze particle distribution"""
        if not particles['areas']:
            return {}
        
        areas = np.array(particles['areas'])
        diameters = np.array(particles['diameters'])
        circularities = np.array(particles['circularities'])
        
        # Convert to real units
        if self.pixel_to_real_ratio:
            areas_real = areas * (self.pixel_to_real_ratio ** 2)
            diameters_real = diameters * self.pixel_to_real_ratio
        else:
            areas_real = areas
            diameters_real = diameters
        
        analysis = {
            'count': len(areas),
            'area_stats': {
                'mean': np.mean(areas_real),
                'std': np.std(areas_real),
                'min': np.min(areas_real),
                'max': np.max(areas_real),
                'median': np.median(areas_real)
            },
            'diameter_stats': {
                'mean': np.mean(diameters_real),
                'std': np.std(diameters_real),
                'min': np.min(diameters_real),
                'max': np.max(diameters_real),
                'median': np.median(diameters_real)
            },
            'circularity_stats': {
                'mean': np.mean(circularities),
                'std': np.std(circularities),
                'min': np.min(circularities),
                'max': np.max(circularities)
            },
            'size_distribution': {
                'small': np.sum(diameters_real < np.percentile(diameters_real, 33)),
                'medium': np.sum((diameters_real >= np.percentile(diameters_real, 33)) & 
                               (diameters_real < np.percentile(diameters_real, 67))),
                'large': np.sum(diameters_real >= np.percentile(diameters_real, 67))
            }
        }
        
        return analysis
    
    def visualize_particle_distribution(self, 
                                      image: np.ndarray, 
                                      particles: Dict, 
                                      analysis: Dict,
                                      save_path: str = 'particle_distribution.jpg',
                                      ocr_boxes: list = None):
        """Visualize particle distribution"""
        # Set font (removed Chinese font dependencies, kept universal fonts)
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Particle Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Original image and detection results
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=12)
        ax1.axis('off')
        
        # Detection results
        ax2 = plt.subplot(2, 4, 2)
        vis_image = image.copy()
        for i, (centroid, bbox) in enumerate(zip(particles['centroids'], particles['bboxes'])):
            y, x = centroid
            y1, x1, y2, x2 = bbox
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(vis_image, (int(x), int(y)), 3, (255, 0, 0), -1)
        ax2.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Detection Results ({len(particles["areas"])} particles)', fontsize=12)
        ax2.axis('off')
        
        # Draw ocr_boxes on original image and detection results
        if ocr_boxes:
            for ocr_box in ocr_boxes:
                ax2.add_patch(
                    plt.Rectangle((ocr_box[0], ocr_box[1]), ocr_box[2]-ocr_box[0], ocr_box[3]-ocr_box[1],
                                  fill=False, edgecolor='purple', linewidth=2, linestyle='--', alpha=0.8))
            print(f"[Visualization] Drawn {len(ocr_boxes)} OCR boxes (purple dashed lines)")
        # Draw all particle bboxes (blue)
        if 'bboxes' in particles:
            for bbox in particles['bboxes']:
                minr, minc, maxr, maxc = bbox
                ax2.add_patch(
                    plt.Rectangle((minc, minr), maxc-minc, maxr-minr,
                                  fill=False, edgecolor='blue', linewidth=1, linestyle=':', alpha=0.7))
            print(f"[Visualization] Drawn {len(particles['bboxes'])} particle bboxes (blue dotted lines)")
        
        # Area distribution histogram
        ax3 = plt.subplot(2, 4, 3)
        if particles['areas']:
            areas = np.array(particles['areas'])
            if self.pixel_to_real_ratio:
                areas = areas * (self.pixel_to_real_ratio ** 2)
            plt.hist(areas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Area ($\\mu m^2$)' if self.pixel_to_real_ratio else 'Area (pixels²)')
            plt.ylabel('Frequency')
            plt.title('Area Distribution')
        
        # Diameter distribution histogram
        ax4 = plt.subplot(2, 4, 4)
        if particles['diameters']:
            diameters = np.array(particles['diameters'])
            if self.pixel_to_real_ratio:
                diameters = diameters * self.pixel_to_real_ratio
            plt.hist(diameters, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel(f'Diameter ({self.unit})' if self.pixel_to_real_ratio else 'Diameter (pixels)')
            plt.ylabel('Frequency')
            plt.title('Diameter Distribution')
        
        # Circularity distribution
        ax5 = plt.subplot(2, 4, 5)
        if particles['circularities']:
            plt.hist(particles['circularities'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.xlabel('Circularity')
            plt.ylabel('Frequency')
            plt.title('Circularity Distribution')
        
        # Size classification pie chart
        ax6 = plt.subplot(2, 4, 6)
        if analysis and 'size_distribution' in analysis:
            sizes = analysis['size_distribution']
            labels = ['Small Particles', 'Medium Particles', 'Large Particles']
            values = [sizes['small'], sizes['medium'], sizes['large']]
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Size Classification')
        
        # Statistical information
        ax7 = plt.subplot(2, 4, 7)
        if analysis:
            stats_text = [
                f"Total Particles: {analysis['count']}",
                f"Average Diameter: {analysis['diameter_stats']['mean']:.2f}",
                f"Diameter Std: {analysis['diameter_stats']['std']:.2f}",
                f"Average Area: {analysis['area_stats']['mean']:.2f}",
                f"Area Std: {analysis['area_stats']['std']:.2f}",
                f"Average Circularity: {analysis['circularity_stats']['mean']:.3f}"
            ]
            if self.pixel_to_real_ratio:
                stats_text = [s + f" {self.unit}" if "Diameter" in s or "Area" in s else s for s in stats_text]
            
            plt.text(0.1, 0.9, '\n'.join(stats_text), transform=ax7.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax7.axis('off')
            ax7.set_title('Statistical Information')
        
        # Scatter plot: Area vs Circularity
        ax8 = plt.subplot(2, 4, 8)
        if particles['areas'] and particles['circularities']:
            areas = np.array(particles['areas'])
            if self.pixel_to_real_ratio:
                areas = areas * (self.pixel_to_real_ratio ** 2)
            plt.scatter(areas, particles['circularities'], alpha=0.6, color='purple')
            plt.xlabel('Area ($\\mu m^2$)' if self.pixel_to_real_ratio else 'Area (pixels²)')
            plt.ylabel('Circularity')
            plt.title('Area vs Circularity')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return vis_image
    
    def analyze_image_with_scale(self, 
                                image: np.ndarray,
                                scale_length_real: float = None,
                                unit: str = "μm",
                                particle_method: str = "watershed",
                                save_path: str = "particle_analysis.jpg",
                                image_path: str = None,
                                access_token: str = None,
                                ocr_boxes: list = None) -> Dict:
        """
        Complete image analysis workflow: scale bar detection + OCR recognition + particle analysis
        """
        print(" Starting image analysis...")
        
        # 1. If there are ocr_boxes, mask these regions first
        if ocr_boxes and len(ocr_boxes) > 0:
            print(f"[mask] Whiting out {len(ocr_boxes)} OCR box regions")
            image = image.copy()
            for box in ocr_boxes:
                x1, y1, x2, y2 = box
                image[y1:y2, x1:x2, :] = 255  # Whiten the area
        
        # 1. Detect scale bar (if detector is available)
        coords = None
        scale_length_pixels = None
        confidence = None
        pixel_length = None
        
        if self.scale_detector is not None:
            print("1. Detecting scale bar...")
            coords, scale_length_pixels, confidence = self.scale_detector.detect_scale(image)
            
            if coords is not None:
                self.scale_length_pixels = scale_length_pixels
                print(f"2. Scale bar detected: {scale_length_pixels:.1f} pixels")
                
                # 2. Auto-recognize physical length with OCR (if image path and access_token are provided)
                if image_path and access_token and scale_length_real is None:
                    print("3. Recognizing scale bar text with OCR...")
                    try:
                        ocr_result = ocr_process(image_path, access_token)
                        print("   Raw OCR result:", ocr_result)
                        
                        words_result = ocr_result.get("words_result", [])
                        for item in words_result:
                            words = item["words"].replace(' ', '')
                            for unit_candidate in ["nm", "μm", "um"]:
                                if unit_candidate in words:
                                    num_str = words.replace(unit_candidate, "")
                                    try:
                                        value = float(num_str)
                                        scale_length_real = value
                                        unit = unit_candidate
                                        print(f"   OCR recognition result: {scale_length_real} {unit}")
                                        break
                                    except:
                                        continue
                            if scale_length_real:
                                break
                    except Exception as e:
                        print(f"   OCR recognition failed: {e}")
                
                # 3. Set scale bar reference
                if scale_length_real:
                    self.set_scale_reference(scale_length_real, unit)
                    self.pixel_to_real_ratio = scale_length_real / scale_length_pixels
                    pixel_length = self.pixel_to_real_ratio
                    # Additional detailed debug output
                    print(f"[Debug] Scale bar pixel length: {scale_length_pixels}")
                    print(f"[Debug] OCR recognized physical length: {scale_length_real} {unit}")
                    print(f"[Debug] Physical length per pixel: {pixel_length} {unit}")
                    print(f"4. Scale bar set: {scale_length_pixels:.1f} pixels = {scale_length_real}{unit}")
                    print(f"   Length per pixel: {pixel_length:.4f}{unit}")
                else:
                    print(f"4. Scale bar detected: {scale_length_pixels:.1f} pixels")
            else:
                print("  No scale bar detected, analysis will use pixel units")
        else:
            print("  No scale bar detector provided, analysis will use pixel units")
        
        if pixel_length is None:
            print("   Length per pixel: Unknown (pixel units only)")
        
        # 5. Detect particles
        print("5. Detecting particles...")
        particles = self.detect_particles(image, method=particle_method, ocr_boxes=ocr_boxes)
        print(f"   Detected {len(particles['areas'])} particles")
        
        # 6. Analyze particle distribution
        print("6. Analyzing particle distribution...")
        analysis = self.analyze_particle_distribution(particles)
        
        # 7. Visualize results
        print("7. Generating visualization...")
        self.visualize_particle_distribution(image, particles, analysis, save_path, ocr_boxes)
        
        # 8. Save detailed data
        self.save_analysis_data(particles, analysis, save_path.replace('.jpg', '_data.csv'), pixel_length, unit)
        
        print(" Analysis completed!")
        
        return {
            'scale_info': {
                'coords': coords,
                'length_pixels': scale_length_pixels,
                'length_real': scale_length_real,
                'confidence': confidence,
                'unit': unit,
                'pixel_length': pixel_length
            },
            'particles': particles,
            'analysis': analysis,
            'pixel_length': pixel_length
        }
    
    def save_analysis_data(self, particles: Dict, analysis: Dict, csv_path: str, pixel_length: float = None, unit: str = "μm"):
        """Save analysis data to CSV file with length per pixel information in the first line"""
        if not particles['areas']:
            return
        
        # Create DataFrame
        data = {
            'Particle_ID': range(1, len(particles['areas']) + 1),
            'Centroid_X': [c[1] for c in particles['centroids']],
            'Centroid_Y': [c[0] for c in particles['centroids']],
            'Area_Pixels': particles['areas'],
            'Diameter_Pixels': particles['diameters'],
            'Circularity': particles['circularities']
        }
        
        if self.pixel_to_real_ratio:
            data['Area_Real'] = [a * (self.pixel_to_real_ratio ** 2) for a in particles['areas']]
            data['Diameter_Real'] = [d * self.pixel_to_real_ratio for d in particles['diameters']]
        
        import pandas as pd
        df = pd.DataFrame(data)
        # Write length per pixel information
        with open(csv_path, 'w', encoding='utf-8') as f:
            if pixel_length:
                f.write(f'# Length per pixel: {pixel_length:.6f}{unit}\n')
            else:
                f.write(f'# Length per pixel: Unknown (pixel units only)\n')
            df.to_csv(f, index=False)
        print(f" Data saved to: {csv_path}")


# Usage example
if __name__ == "__main__":
    # Load scale bar detector
    scale_detector = LightweightScaleDetectorWithPreprocessing(
        model_path='lightweight_scale_detector.pth',
        input_size=(320, 320),
        hidden_dim=32
    )
    
    # Create particle analyzer
    analyzer = ParticleAnalyzer(scale_detector)
    
    # Create test image (containing particles and scale bar)
    test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    
    # Add scale bar
    cv2.line(test_image, (50, 50), (250, 50), (255, 255, 255), 5)
    
    # Add some simulated particles
    for i in range(20):
        x = np.random.randint(60, 280)
        y = np.random.randint(60, 280)
        radius = np.random.randint(5, 15)
        cv2.circle(test_image, (x, y), radius, (200, 200, 200), -1)
    
    # Analyze image
    results = analyzer.analyze_image_with_scale(
        test_image,
        scale_length_real=100,  # Assume scale bar represents 100 micrometers
        unit="μm",
        particle_method="watershed"
    )
    
    print(" Particle analysis completed!")
