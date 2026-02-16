# Particle Analysis Project

This project integrates the MU-KAN semantic segmentation model and YOLOV11 object detection model for particle image analysis, including scale bar detection, OCR recognition, and particle feature extraction.

## File Description

- `particle_analysis_demo.py` - Main particle analysis script that integrates all functionalities
- `view_results.py` - Result visualization script
- `scale_detector.py` - Scale bar detector
- `particle_analyzer.py` - Particle analyzer
- `mock_ocr.py` - OCR simulator
- `scale_detector_kan.pth` - Pre-trained weights for scale bar detection model
- `Particle Analysis Usage Instructions.md` - Detailed usage guidelines


## Key Features

1.Multi-model Support: MU-KAN semantic segmentation + YOLO object detection
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

1.Ensure MU-KAN model weight files are placed in the correct directory
2.The scale bar detection model is pre-trained and ready for direct use
3.Supports multiple image formats: jpg, png, bmp, etc.
4.Analysis results are saved as CSV files and visual charts
