# Tomato leaf disease prediction realtime
## Dataset
This dataset is about tomato leaf disease and I think it is a sample of some datasets, This dataset has 10 classes and each class contains 1000 images for training and 100 for validation, but at first, I separated 100 images from the train and created test folder and move images to it. 
The dataset is available in this [link](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf).

## Model
This model has a simple Convolutional Neural Network that we implement with TensorFlow and it gets images with the size of 128*128*3. We train data with just 40 epochs and have a 97% accuracy in training and 94% in testing and it just needs to be trained with more epochs.
The model size is about 3 megabytes and if you convert it to tf-lite the size of the model will decrease you can use it in embedded systems such as Raspberry Pi and Jetson Nano for real-time and this is the novelty of this project than similar projects. According to my experience, the inference of the model in Raspberry Pi 4 is less than 200 milliseconds.
## Result 
<img src="https://github.com/Amahseyn/Tomato-leaf-disease-prediction/blob/main/acc.png" align="center" height="300" width="400"/>
<img src="https://github.com/Amahseyn/Tomato-leaf-disease-prediction/blob/main/loss.png" align="center" height="300" width="400"/>
