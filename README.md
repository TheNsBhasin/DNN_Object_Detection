# DNN_Object_Detection

This code demonstrates usage of OpenCV deep learning module (dnn module) with MobileNet-SSD network for object detection.

## Setup

### Dependencies

```Linux
pip install -r requirements.txt
```

### Execution
```Linux
python dnn_object_detection.py \
--prototxt MobileNetSSD_deploy.prototxt.txt \
--model MobileNetSSD_deploy.caffemodel \
--labels object_detection_classes_pascal_voc.txt 
```

### Usage

```Linux
python dnn_object_detection.py [-h] -p PROTOTXT -m MODEL -l LABELS
                               [-c CONFIDENCE] [-v VIDEO]
```

## Description

Object recognition is a computer vision technique for identifying objects in images or videos. Object recognition is a key output of deep learning and machine learning algorithms.

As part of Opencv 3.4.+ deep neural network(dnn) module was included officially. The dnn module allows load pre-trained models from most populars deep learning frameworks, including Tensorflow, Caffe, Darknet, Torch. Besides MobileNet-SDD other architectures are compatible with OpenCV 3.4.1 :

* GoogleLeNet
* YOLO
* SqueezeNet
* Faster R-CNN
* ResNet

This API is compatible with C++ and Python.

### Implementation

Network used - [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)

We can therefore detect 20 objects in images (+1 for the background class), including airplanes, bicycles, birds, boats, bottles, buses, cars, cats, chairs, cows, dining tables, dogs, horses, motorbikes, people, potted plants, sheep, sofas, trains, and tv monitors.

## Results

![Alt Text](https://github.com/TheNsBhasin/DNN_Object_Detection/blob/master/output.gif)

![Alt Text](https://github.com/TheNsBhasin/DNN_Object_Detection/blob/master/sample.jpeg)

## References

[Real-time object detection with deep learning and OpenCV](https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/)

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)

[OpenCV deep learning module](https://github.com/opencv/opencv/tree/master/samples/dnn)
