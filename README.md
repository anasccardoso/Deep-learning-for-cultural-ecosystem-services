# Deep-learning-for-cultural-ecosystem-services

This project was implemented using two different convolutional neural networks architectures (ResNet152 and VGG16), as well as three different weights (Places365, ImageNet and our own weights/weights from scratch).
In the code files there is only represented the ResNet152 architecture with ImageNet weights, since this was the pair that achieved the best results in our project.
All of the code was run using Google Colab. 

If you want to implement the VGG16 architecture in the "nature_vs_human.py", "multilabel_classification.py" and "transferability_generalization.py" files, substitute "from keras.applications.resnet import ResNet152" for "from keras.applications.vgg16 import VGG16" (line 14), as well as "ResNet152()" for "VGG16()" (line 77). If you change ResNet152 for VGG16, make sure to change the learning rate (lr) to 0.000001 (line 80).
To use the Places365 weights in the ResNet152 architecture, substitute "model.add(ResNet152(include_top = False, pooling = "max", input_shape = (173, 273, 3), weights = "imagenet"))" for "resnet_model = ResNet152(include_top = False, pooling = "max", input_shape = (173, 273, 3))"
                                                                                                                                                             "resnet_model.load_weights("/content/resnet152_places365.h5", by_name = True)"
                                                                                                                                                             "model.add(resnet_model)"  
To use the Places365 weights in the VGG16 architecture, add "from vgg16_places_365 import VGG16_Places365" and substitute "weights = "imagenet"" for "weights = places" (line 77).
To use our own weights/weights from scratch in both architectures, substitute "weights = "imagenet"" for "weights = None" (line 77).
