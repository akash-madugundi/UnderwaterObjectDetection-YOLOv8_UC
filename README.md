# Underwater Object Detection- YOLOv8-UC

Underwater object detection presents unique challenges due to *light attenuation*, *scattering*, and *complex environments*, which often result in reduced detection accuracy. This paper presents the implementation and analysis of YOLOv8-UC, an improved underwater object detection algorithm based on YOLOv8.

---

## Architecture
- A modified Dilation-wise Residual **(DWR)** C2f module to expand the receptive field for improved feature extraction.
- Large Separable Kernel Attention **(LSKA)** integration with SPPF to reduce information loss during feature fusion.
- A redesigned detection head utilizing **RepConv** to create a shared parameter structure.
- an **Inner-SIoU** loss function that employs auxiliary bounding boxes at different scales for improved bounding box regression.

## Highlights
- **Regional Residualization:** This step generates concise feature maps with different region sizes, using 3×3 convolutions combined with batch normalization and ReLU activation.
- **Semantic Residualization:** Multi-rate dilated depth convolutions perform morphological filtering on the regional features, applying specific receptive fields to each channel.
- **Computational Cost:** While testing, the computational cost is reduced by using enhanced RepConv.
  
## Datasets:
- [URPC2019](https://universe.roboflow.com/underwater-fish-f6cri/urpc2019-nrbk1/dataset/3/download/yolov8) - ~4,000 images
- [Underwater Plastic](https://www.kaggle.com/datasets/arnavs19/underwater-plastic-pollution-detection) - ~14,000 images

---

## Installation & Setup
#### Clone the Repository
```bash
git clone <repository-url>
cd YOLOv8-UC_UnderwaterDetection
```

#### Install dependencies:
```
pip install -r requirements.txt
```

#### Training Model:
```bash
python -m ultralytics.yolo.v8.detect.train model=/ultralytics/models/v8/yolov8n.yaml data=ultralytics/yolo/data/datasets/<your_yaml>.yaml epochs=50 imgsz=640
```

#### Prediction:
```
python -m ultralytics.yolo.v8.detect.predict model=/runs/detect/train<xx>/weights/best.pt source=<img_path>
```
