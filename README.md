# [Image Segmentation | fastai Unet](https://github.com/billumillu/Image-Segmentation-fastai-Unet)
This implementation uses fastai's UNet model, where the CNN backbone (e.g. ResNet) is pre-trained on ImageNet and hence can be fine-tuned with only small amounts of annotated training examples.
## Data
In this notebook, we use a toy dataset specified in DATA_PATH which consists of 129 images of 4 classes of beverage containers {can, carton, milk bottle, water bottle}. For each image, a pixel-wise ground-truth mask is provided for training and evaluation.
## Model Performance
![](./images/seg_output1.jpg)
## Test Image
![](./images/seg_output.jpg)

# [Object Detection | FasterRCNN](https://github.com/billumillu/Object-Detection-FasterRCNN)
This repository uses torchvision's Faster R-CNN implementation which has been shown to work well on a wide variety of Computer Vision problems.
For the DetectionLearner, we use Faster R-CNN as the default model, and Stochastic Gradient Descent as our default optimizer.
Our Faster R-CNN model is pretrained on COCO, a large-scale object detection, segmentation, and captioning dataset that contains over 200K labeled images with over 80 label cateogories.
## Data
In this notebook, we use a toy dataset called Fridge Objects, which consists of 134 images of 4 classes of beverage container {can, carton, milk bottle, water bottle} photos taken on different backgrounds.
## Output
![](./images/det_output.jpg)
