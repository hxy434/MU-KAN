# Particle Analysis Project

This project integrates the UKAN semantic segmentation model and YOLO object detection model for particle image analysis, including scale bar detection, OCR recognition, and particle feature extraction.

## File Description

- `particle_analysis_demo.py` - Main particle analysis script that integrates all functionalities
- `ukan_inference.py` - UKAN model inference script
- `view_results.py` - Result visualization script
- `scale_detector.py` - Scale bar detector
- `particle_analyzer.py` - Particle analyzer
- `mock_ocr.py` - OCR simulator
- `scale_detector_kan.pth` - Pre-trained weights for scale bar detection model
- `Particle Analysis Usage Instructions.md` - Detailed usage guidelines

## Usage

### 1. Run Particle Analysis
```bash
python particle_analysis_demo.py --image path/to/your/image.jpg --method ukan

### 2. View UKAN Inference Results
```bash
python view_results.py --image path/to/your/image.jpg
```

### 3. Run Only UKAN Inference
```bash
python ukan_inference.py --image path/to/your/image.jpg
```

Detection Method Selection
--method ukan: Use UKAN semantic segmentation model
--method yolo: Use YOLO object detection model
--method auto: Auto-selection (UKAN is used by default)

## Key Features

1.Multi-model Support: UKAN semantic segmentation + YOLO object detection
2.Scale Bar Detection: Automatically detect scale bars in images
3.OCR Recognition: Identify numerical values on scale bars
4.Particle Analysis: Extract particle features (area, diameter, circularity, etc.)
5.Result Visualization: Generate analytical result charts
6.Physical Measurement: Convert pixel measurements to actual physical dimensions

## Dependencies

- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Albumentations
- Ultralytics (YOLO)
- EasyOCR (可选，用于真实OCR)

## Notes

1.Ensure UKAN model weight files are placed in the correct directory
2.The scale bar detection model is pre-trained and ready for direct use
3.Supports multiple image formats: jpg, png, bmp, etc.
4.Analysis results are saved as CSV files and visual charts
