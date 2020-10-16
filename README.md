# Computer Vision Projects

# [Image Segmentation | fastai Unet](https://github.com/billumillu/Image-Segmentation-fastai-Unet)
This implementation uses fastai's UNet model, where the CNN backbone (e.g. ResNet) is pre-trained on ImageNet and hence can be fine-tuned with only small amounts of annotated training examples.
## Data
In this notebook, we use a toy dataset specified in DATA_PATH which consists of 129 images of 4 classes of beverage containers {can, carton, milk bottle, water bottle}. For each image, a pixel-wise ground-truth mask is provided for training and evaluation.
## Model Performance
![](/images/seg_output1.jpg)
## Test Image
![](/images/seg_output.jpg)

# [Object Detection | FasterRCNN](https://github.com/billumillu/Object-Detection-FasterRCNN)
This repository uses torchvision's Faster R-CNN implementation which has been shown to work well on a wide variety of Computer Vision problems.
For the DetectionLearner, we use Faster R-CNN as the default model, and Stochastic Gradient Descent as our default optimizer.
Our Faster R-CNN model is pretrained on COCO, a large-scale object detection, segmentation, and captioning dataset that contains over 200K labeled images with over 80 label cateogories.
## Data
In this notebook, we use a toy dataset called Fridge Objects, which consists of 134 images of 4 classes of beverage container {can, carton, milk bottle, water bottle} photos taken on different backgrounds.
## Output
![](/images/det_output.jpg)

# [Pose Estimation | pytorch](https://github.com/billumillu/Pose-Estimation-pytorch)
I used a custom pre-trained model. It was trained on the COCO dataset (250,000 people with keypoints - 42.7GB!).

My goal for this project was to get a simple model that will give me the pose for a human body. I plan to later use this in another project. The idea is to classify people based on their pose into - attackers, victims. This could help in preventing/controlling crimes. I currently need to learn about video analytics, and collect data (crime videos) for this project. Then, I will integrate it with this model.
## Output
![](/images/pose.jpg)

# [Face Verification & Recognition | FaceNet](https://github.com/billumillu/Face-Verification-and-Recognition)

Face recognition problems commonly fall into two categories:

* Face Verification - "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem.
* Face Recognition - "who is this person?". For example, employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem.

FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person. We use FaceNet's pre-trained model.

## Data
Own database consisting of 12 individuals, an image for each.

## Outputs
camera_0.jpg is a picture of Younes.

![](/images/camera_0.jpg)
### Face Verification
![](/images/face_ver.jpg)

Gives the output as 'True' or 'False'
### Face Recognition
![](/images/face_rec.jpg)

Gives the name of the person.

# [Car Detection for Autonomous Driving | YOLOv2, tensorflow 1.x](https://github.com/billumillu/Car-Detection-for-Autonomous-Driving)

YOLO ("you only look once") is a popular algoritm because it achieves high accuracy while also being able to run in real-time. Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes. We are loading an existing pretrained Keras YOLO model stored in "yolo.h5".

## Data
This project was done on Coursera's Deep Learning Specialization. To collect data, they mounted a camera to the hood (meaning the front) of the car, which takes pictures of the road ahead every few seconds while you drive around. They gathered all these images into a folder and labelled them by drawing bounding boxes around every car they found.

## Output
![](/images/car_bbox.jpg)
