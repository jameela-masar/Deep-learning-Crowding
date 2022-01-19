
## Abstract:
The goal of the project is to manage crowds at Almasjed Alharam by training a model that counts the crowds in crowded places and then helps us with several things, including preventing crowds and facilitating the evacuation process in the event of a fire. We used the CSRNet model, which is Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes and ShanghaiTech dataset in the training process, and finally, Deployment Using Flask.

## Design:
This project arose from the initiative of SDAIA harnessing digital products in the service of the guests of the holy places. The number of people is measured by the average distance per head and this will help to estimate the number of people in crowded places and is considered a very important topic in artificial intelligence, because it Applying a high-accuracy model to count people in a crowd will help us move forward with the application of AI for crowd management.

## Data:
A ShanghaiTech dataset was used and consisted of two parts A and B: Part A consists of images containing high density of the crowd, and Part B contains images of low density of the crowd.
In the future, we hope to work on training the model on a data set of Almasjed Alharam.

## Algorithms:
Preprocessing:
1198 annotated images were used. Firstly, we produced the ground truth, and then we produced the density map for each image.
Model:
A CSRnet model was used. This model is used at the front end of the VGG-16, and at the back end, dilated convolutional layers are used.
Model Evaluation and Selection:
As previously mentioned, the data set is divided into two parts A and B, and each part is divided into Train and Test.
Activation function : relu
Pooling layer : MaxPooling
Optimizer : sgd
Loss : Euclidean distance loss 
Metrics = MSA (mean squared error )
Epoch = 700 
Result :
Validation MAE =  43.378
Test MAE = 65.927

## Tools:
* Programs:Jupyter Notebook , Flask
* Libraries: keras-tensorflow,h5py, scipy, PIL, numpy, os, glob, matplotlib, json, tqdm, cv2, random, math, sys, pandas, sklearn, future, re, flask, werkzeug, gevent.
* Functions: gaussian_filter_density, array, resize, expand_dims, shape, random, save_weights, to_json, load_weights, compile, summary, join, replace, predict, mean_absolute_error.
* Plots: pointplot, barplot, histogram, scatter, distplot, Pie Chart.

## Communication:
* Present a 5-minute slide presentation and a one-page report summarizing the project, beginning with an abstract and detailing the project, including the five major components. Also, my project will be included on my GitHub page.
* At the end of this project, by researching and comparing several models, we came up with a model capable of counting crowds in crowded places with good accuracy.

![](https://user-images.githubusercontent.com/71223849/150178852-b57afa88-7407-454c-b15f-cdc531fd63c9.png)

## Note:
To apply the demo of the project, run the app.py file.
