# Tomato leaf disease prediction realtime
### Dataset
This dataset is about tomato leaf disease prediction and I think it is a sample of some datasets, This dataset has 10 classes and each class contains 1000 images for training and 100 for validation, but at first I separated 100 images from the train and created test folder and move images to it. 
The dataset is available in this [link](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)
### Model
This model has a simple Convolutional Neural Network that we implement with TensorFlow and it gets images with the size of 128*128*3. We train data with just 40 epochs and have a 97 accuracy in training and 94 in testing and it just needs train with more epochs.
### Result 
<img src="https://github.com/Amahseyn/Tomato-leaf-disease-prediction/blob/main/acc.png" align="center" height="350" width="600"/>
<img src="https://github.com/Amahseyn/Tomato-leaf-disease-prediction/blob/main/loss.png" align="center" height="350" width="600"/>
