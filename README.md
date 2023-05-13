# Emojimation

---
Facial expression recognition 

IEEE Envision Project

year: 2023

---



### Mentors

- Vartika Rao

- Aman Raj

### Members

- Smruthi Bhat

- Krishna Tulsyan

- Chirag S

- Bhuvanesh Singla

## Aim

This project aims to build a deep-learning model to classify human facial expressions.

## Introduction

Images can both express and affect people's emotions. It is intriguing and important to understand what emotions are conveyed and how they are implied by the visual content of images. Inspired by advancements in computer vision deep convolutional neural networks (CNN) in visual recognition, we will classify the images into the following categories:
 0:angry
 1:disgust
 2:fear
 3:happy
 4:sad
 5:surprise
 6:natural
 
We will use the Keras library to build the deep learning model and the FER2013 dataset (facial expression recognition) to train and test our model.

The implementation of the model is done using a GUI interface and a web app using Streamlit.

## Libraries Used 
- Pandas 
- Numpy 
- Matplotlib 
- Tensorflow->Keras
- cv2
- PIL
- Streamlit

## Datasets 

###  Facial Expression Recognition 2013

The data consists of 48x48 pixel grayscale images of faces. There are a total of 28,709 images. 

### Bar graph representing the number of images of each emotion: 

![image](https://github.com/amanrajNitk/image/blob/6f2106652508873d5685346314a6eb1744738a23/Screenshot%202023-05-13%20235406.png)

## Convolutional Neural Network

CNNs or convolutional neural nets are a type of deep learning algorithm that does really well at learning images. That’s because they can learn patterns that are translation invariant and have spatial hierarchies. 

By leveraging this power of CNN, it could also be used to classify audio clips. We can extract features which look like images and then shape them in a way in order to feed them into a CNN. 

That’s exactly what’s used in our project, extracting audio features and then shaping them into a multi-dimensional matrix, which is then fed into the CNN for training. This builds a robust model which is capable of classifying the emotions of an audio clip. 

![image5](https://user-images.githubusercontent.com/78913275/175575885-c2503de9-14e7-451d-bdb5-b80f533009d0.png)

## Model Architecture

We have used an alternate sequence of Convolutional Layers and MaxPooling Layers for our Model. Our model also includes other layers like 

Dropout->it randomly ignores a set of neurons in the model in order to reduce its complexity and also helps reduce overfitting. 
Flatten-> it converts the output from the Convolutional and MaxPooling Layers into a 1-dimensional array for inputting it to the next layer. 
Dense-> it was used as the output layer for classifying the emotion of the audio clip. 

ReLU is the activation function used for all Convolutional Layers. 

We have also used Softmax as the activation function for the final layer as our model predicts a multinomial probability distribution. 

Using ReduceLROnPlateau helps us monitor the training loss and if no improvement is seen for a patience number of epochs, the learning rate is reduced by a certain factor. 

The learning rate was initially set to 0.001 and was adjusted according to ReduceLROnPlateau throughout the process of training. 

The loss function used in this model is categorical_crossentropy. 

Adam optimizer was used along with a batch size of 32 and 200 epochs.  The above Hyperparameters could still be slightly tweaked to further improve accuracy. 

The kernel size used in the convolutional layers are either 3 or 5 and the pool size in the maxpooling layers are all set to 3 making strides of 2. 

## Result 
<img width="402" alt="image" src="https://github.com/Chiru2004/Emojimation/assets/123258424/0d8a8f1b-f4f7-4a2b-b581-e44b22853b17">
<br>Training and validation accuracy

<br>
On the test set, we have achieved an accuracy of around 65% and an accuracy of around 68% on the validation set. 
We are trying to improve the accuracy of our model. 

## A Confusion matrix of the predicted output is shown below 

<img width="270" alt="image" src="https://github.com/Chiru2004/Emojimation/assets/123258424/94c65447-90c1-4855-a565-4c88bc6fa4c4">

## Conclusion 

This Project gave us an oppurtunity to learn image processing and classification of images based on emotions through deep learning implemented through Tensorflow Keras. It provided proper insight over feature extraction from images through CNNs. Understanding the CNN architecture to obtain optimum accuracy and minimum loss was exciting. Development of GUI using openCV and Tkinter which is able to classify the face and output the emotion was definetly a plus to knowledge learnt in the project.

## References

1. https://in.coursera.org/learn/neural-networks-deep-learning	

2. https://in.coursera.org/specializations/machine-learning-introduction#courses

3. https://github.com/marinavillaschi/ML-AndrewNg

4. https://github.com/amanchadha/coursera-deep-learning-specialization

5. https://deeplizard.com/resource/pavq7noze2.

6. https://deeplizard.com/resource/pavq7noze3 
 
