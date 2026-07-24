# MU-KAN: An End-to-End Framework for Robust Nanoparticle Segmentation and Automated Size Measurement in SEM Images
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the RSC Advances paper **Automated SEM-Based Nanoparticle Metrology for Materials Characterization via Segmentation and Robust Scale-Bar Recognition**. MU-KAN is a Multi-scale U-Net Kolmogorov–Arnold Network tailored for accurate nanoparticle boundary extraction in SEM images under challenging conditions (severe noise, low contrast, particle overlap/agglomeration), integrated with an automatic scale recovery module for pixel-to-physical unit conversion and automated particle size distribution analysis.

## Key Features
- **Robust Semantic Segmentation**: Fuses Moga Block-based multi-scale feature encoding and KAN-driven nonlinear modeling to boost boundary detection accuracy for noisy, low-contrast, and overlapped nanoparticle SEM images.
- **Automatic Scale Recovery**: Combines YOLOv11-based scale-bar localization and OCR-based annotation recognition to realize accurate pixel-to-physical (nm/μm) unit conversion with ultra-low relative error.
- **End-to-End Quantitative Analysis**: Automatically computes particle size distribution, equivalent circular diameter, circularity, and other morphological metrics from raw SEM images without manual intervention.
- **Public Benchmark Dataset**: A manually annotated dataset of 370 SEM images with scale-bar ground truth (covering extreme imaging cases like low contrast and severe blur) is released for scale recognition evaluation.
- **User-Friendly Web Service**: An interactive web application supporting single/batch image processing, real-time visualization, and statistical report export (no local environment configuration required).
- **State-of-the-Art Performance**: Achieves average F1-scores of 0.9630 (NanoSEM-464) and 0.9029 (NanoSEM-1707), significantly outperforming mainstream segmentation methods (U-KAN, DeepLabv3+, TransUNet, etc.).



## Main Contributions
1. Propose MU-KAN, one of the first attempts to integrate KANs with multi-scale feature encoding for quantitative nanoparticle segmentation in SEM imaging, substantially improving boundary robustness for noisy, low-contrast, and highly overlapped particles.
2. Release a benchmark dataset of 370 multi-scenario SEM images with scale-bar ground truth, providing a standardized resource for nanoparticle scale recognition evaluation.
3. Establish a fully automated closed-loop pipeline from raw SEM image input to particle segmentation, physical scale calibration, size estimation, and distribution reporting, enabling high-throughput nanoparticle characterization.
4. Develop a robust scale recovery strategy combining YOLOv11 and OCR, achieving accurate pixel-to-physical conversion with a relative error of only 3.8604% and strong adaptability to diverse imaging artifacts.

## Framework Overview
The MU-KAN framework consists of three core modular components, forming a seamless end-to-end workflow for SEM nanoparticle analysis:
1. **MU-KAN Segmentation Network**: U-Net-based architecture integrated with Moga Block (multi-scale feature interaction) and Tokenized KAN Block (nonlinear boundary modeling) for high-precision semantic segmentation.
2. **Scale Recovery Module**: YOLOv11 for accurate scale-bar localization + OCR for unit/value recognition, realizing reliable pixel-to-physical unit conversion across diverse SEM image settings.
3. **Intelligent Statistical Analysis Module**: Automatically calculates particle morphological metrics (area, equivalent diameter, circularity) and generates statistical distributions (histograms, scatter plots) and quantitative analysis reports.

## Installation
### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)
- Other dependencies (listed in `requirements.txt`)

### Step 1: Clone the Repository
```bash
git clone https://github.com/hxy434/MU-KAN.git
cd MU-KAN
```

### Step 2: Create Conda Environment (Recommended)
```bash
conda create -n mukan python=3.9
conda activate mukan
```

### Step 3: Install Dependencies
```bash
# Install PyTorch matching your CUDA version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install other required packages
pip install -r requirements.txt
```

### Step 4: Download Datasets
- **Public Benchmarks**: NanoSEM-464 and NanoSEM-1707 datasets are available in the project's data directory.
- **MU-KAN Scale Bar Dataset**: 370 annotated SEM images with scale-bar ground truth are released in `data/scale_bar_dataset/`.
- Unzip all datasets and place them in the `data/` directory following the structure in `data/README.md`.

## Quick Start
### 1. Model Training
Train the MU-KAN segmentation model on NanoSEM-464/NanoSEM-1707 datasets:
```bash
# Train on NanoSEM-464
python train.py --dataset nanosem464 --epochs 200 --batch_size 8 --lr 3e-4 --gpu 0

# Train on NanoSEM-1707
python train.py --dataset nanosem1707 --epochs 300 --batch_size 16 --lr 4e-4 --gpu 0
```
Adjust training configurations (learning rate, batch size, etc.) in `config/train_config.py`.

### 2. Inference (Single/Batch Image)
Run segmentation and scale recovery on raw SEM images:
```bash
# Single image inference
python infer.py --img_path ./examples/SEM_sample.png --save_path ./results --gpu 0

# Batch image inference
python infer.py --img_dir ./examples/ --save_path ./results --gpu 0
```
**Output Results**:
- Segmentation mask, probability map, and overlay visualization
- Scale bar detection/recognition results (pixel length, physical unit/value)
- Quantitative particle metrics (count, average diameter, average circularity, etc.)
- Statistical plots (area/diameter/circularity distribution, area-diameter scatter plot)
- Comprehensive analysis report in PDF/CSV format

### 3. Web Service Deployment
Launch the interactive web application for easy SEM image analysis:
```bash
# Start the web server (default port: 8080)
python web/app.py --port 8080
```
Open your browser and visit `http://localhost:8080` to access the web interface, which supports image upload, real-time processing, visualization, and result export.

## Dataset Details
### Public Segmentation Benchmarks
| Dataset       | Number of Images | Image Size | Train/Val/Test Split | Main Task |
|---------------|------------------|------------|---------------------|-----------|
| NanoSEM-464   | 464              | 256×256    | 8:2 (371/93)        | Segmentation |
| NanoSEM-1707  | 1707             | 256×256    | 8:1:1 (1366/171/170)| Segmentation & Generalization |

### MU-KAN Scale Bar Dataset
- **370 SEM images** with manually annotated scale-bar ground truth (center coordinates, pixel length, physical unit/value).
- Covers extreme imaging scenarios: low contrast, severe blur, diverse scale bar positions (corner/floating), units (nm/μm), and resolutions (512×512, 1024×1024).
- Annotations are stored in JSON format in `data/scale_bar_dataset/annotations/`.

## Evaluation Metrics
### Segmentation Metrics
Standard semantic segmentation metrics with a smoothing term ε (to avoid division-by-zero errors) are used:
- **IoU (Intersection over Union)**: Core metric for segmentation accuracy, measuring the overlap between predicted and ground-truth masks.
- **F1-Score**: Harmonic mean of Precision and Recall, balancing false positives and false negatives (ideal for small-object segmentation).
- **Accuracy**: Overall pixel-level prediction correctness.
- **Precision**: Proportion of correctly predicted foreground pixels.
- **Dice Loss**: Loss function for model training (1 - Dice Coefficient).

### Scale Recovery Metric
- **Relative Error Percentage**: Defined as `(predicted scale length / ground-truth scale length) × 100%`, the key metric for scale bar recognition performance.

## Experimental Results
### 1. Segmentation Performance (F1-Score)
| Dataset       | MU-KAN (Ours) | U-KAN  | DeepLabv3+ | TransUNet | FPN    |
|---------------|---------------|--------|------------|-----------|--------|
| NanoSEM-464   | 0.9630        | 0.9476 | 0.8784     | 0.8472    | 0.8684 |
| NanoSEM-1707  | 0.9029        | 0.8821 | 0.7969     | 0.8767    | 0.7722 |

### 2. Scale Recovery Performance
| Method         | Relative Error (%) |
|----------------|--------------------|
| U-Net (ResNet) | 65.3562            |
| ViT            | 33.9197            |
| YOLOv11 (Ours) | 3.8604             |

## Directory Structure
```
MU-KAN/
├── config/          # Training/inference/web configuration files
├── data/            # Datasets and annotations (with detailed README)
├── docs/            # Framework figures, web interface screenshots, and tutorials
├── examples/        # Sample SEM images for quick test and demonstration
├── models/          # Model implementations
│   ├── mukan/       # MU-KAN segmentation network (CB, MB, TKB modules)
│   ├── yolov11/     # YOLOv11 for scale-bar localization
│   └── baselines/   # Baseline models (U-KAN, DeepLabv3+, TransUNet, etc.)
├── scale_recovery/  # Scale bar detection + OCR recognition module
├── stats/           # Statistical analysis and visualization code
├── web/             # Web service (frontend SPA + backend microservices)
├── train.py         # Model training script
├── infer.py         # Inference script for single/batch images
├── requirements.txt # List of dependent packages
└── README.md        # Project documentation
```

## Pretrained Models
Pretrained weights for the MU-KAN segmentation network and YOLOv11 scale-bar detection model are available in the `models/pretrained/` directory, which can be directly used for inference and web service deployment without retraining.

## Limitations & Future Work
### Current Limitations
1. Instance-level separation of highly agglomerated nanoparticles is not fully achieved with pure semantic segmentation.
2. OCR recognition may fail if the scale bar is severely occluded or overlaps with particle regions.
3. Boundary localization accuracy may degrade in SEM images with extremely complex backgrounds.

### Future Work
1. Integrate instance segmentation with topological constraints to realize accurate separation of highly agglomerated nanoparticles.
2. Develop a more robust scale recognition model for occluded/blurred scale bars based on multimodal fusion.
3. Add adaptive feature enhancement modules for weakly structured regions in SEM images to further improve boundary detection accuracy.
4. Extend the framework to support multimodal imaging data (TEM, AFM) for universal nanomaterial characterization.
5. Implement online incremental learning to adapt the model to custom nanoparticle datasets with minimal annotation.

## Acknowledgements
This work was supported by:
- Natural Science Foundation of Anhui Province (No.2508085MF153, No.JZ2025AKZR0608)
- Hefei Municipal Natural Science Foundation (No.HZR2403, No.JZ2024HKZR0706, No.W2024JSKF0762)
- General Project of Anhui Province Outstanding Young Teachers Cultivation Program in 2024 (No.YQYB2024096)

We thank the authors of U-Net, KAN, YOLOv11, and other open-source projects for their valuable code and model contributions.

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Contact
For questions, issues, or collaboration requests, please contact the project maintainers via the GitHub Issues page or the official email address in the paper.

Issues and pull requests are warmly welcome!
