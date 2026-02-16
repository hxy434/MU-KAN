# 粒子分析项目

这个项目集成了UKAN语义分割模型和YOLO目标检测模型，用于粒子图像的分析，包括比例尺检测、OCR识别和粒子特征提取。

## 文件说明

- `particle_analysis_demo.py` - 主要的粒子分析脚本，集成了所有功能
- `ukan_inference.py` - UKAN模型推理脚本
- `view_results.py` - 结果可视化脚本
- `scale_detector.py` - 比例尺检测器
- `particle_analyzer.py` - 粒子分析器
- `mock_ocr.py` - OCR模拟器
- `scale_detector_kan.pth` - 比例尺检测模型权重
- `粒子分析使用说明.md` - 详细使用说明

## 使用方法

### 1. 运行粒子分析
```bash
python particle_analysis_demo.py --image path/to/your/image.jpg --method ukan
```

### 2. 查看UKAN推理结果
```bash
python view_results.py --image path/to/your/image.jpg
```

### 3. 仅运行UKAN推理
```bash
python ukan_inference.py --image path/to/your/image.jpg
```

## 检测方法选择

- `--method ukan`: 使用UKAN语义分割模型
- `--method yolo`: 使用YOLO目标检测模型
- `--method auto`: 自动选择（默认使用UKAN）

## 功能特性

1. **多模型支持**: UKAN语义分割 + YOLO目标检测
2. **比例尺检测**: 自动检测图像中的比例尺
3. **OCR识别**: 识别比例尺上的数值
4. **粒子分析**: 提取粒子的面积、直径、圆度等特征
5. **结果可视化**: 生成分析结果图表
6. **物理测量**: 将像素测量转换为实际物理尺寸

## 依赖要求

- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Albumentations
- Ultralytics (YOLO)
- EasyOCR (可选，用于真实OCR)

## 注意事项

1. 确保UKAN模型权重文件在正确位置
2. 比例尺检测模型已预训练，可直接使用
3. 支持多种图像格式：jpg, png, bmp等
4. 分析结果会保存为CSV文件和可视化图表 