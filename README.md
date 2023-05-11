# Emojimation
---
# layout: post
title: "Emojimation"
description: "Facial expression recognition "
categories: IEEE Envision
# thumbnail: "filename.jpg"
year: 2023
---

### Project Guide

- _____________

### Mentors

- Vartika Rao

- Aman Raj

### Members

- Smruthi Bhat

- Krishna Tulsyan

- Chirag S

- Bhuvanesh Singla

## Acknowledgments
We had a great experience learning about image processing and computer vision using deep learning models. we express our deep gratitude to Dr. __________ for allowing us to work on this project. 

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

###  Facial Expression Recognition

The data consists of 48x48 pixel grayscale images of faces. It consists of 28,709 images. 

There are a set of 200 target words spoken by two female speakers, resulting in 2800 data points in total. 

### Bar graph representing the number of images in each emotion: 

<!-- ![image1](https://user-images.githubusercontent.com/78913275/175575200-3846ff05-9664-4be5-88c0-751710f48246.png) -->


## Data Augmentation 

A key part of deep learning is to feed the neural network with a lot of data so that it can learn to generalize well. Data Augmentation is used to generate additional audio file samples by slightly modifying already existing data. This helps us to minimize overfitting of our model. 

### Noise injection

In this process we add white noise to an audio sample, hence producing additional audio samples having slightly different audio features but representing the same emotion. 

### Time stretching

Time stretching is the process of changing the speed or duration of an audio signal without affecting its pitch. 

### Pitch scaling

It is the process of changing the pitch without affecting the speed. It should only be used to a small extent as pitch forms an important part of emotion. 

## Feature Extraction 

The following features have been extracted from each audio sample. These set of features helps the model distinguish each audio sample from the another. 

### Zero Crossing Rate

The zero-crossing rate is the rate at which a given signal changes from positive to zero to negative or vice versa. 

### Chroma

Chroma and chroma related features are a powerful tool for analyzing music whose pitches can be meaningfully categorized (often into twelve categories) and whose tuning approximates to the equal-tempered scale. 

### Mel-Frequency Cepstral Coefficients

The Mel-Frequency Cepstrum(MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. Mel-frequency cepstral coefficients are coefficients that collectively make up an MFC. 

### Root Mean Square Energy

RMS is a meaningful way of calculating the average of values over a period of time. With audio, the signal value (amplitude) is squared, averaged over a period of time, then the square root of the result is calculated. 

### Mel Spectrogram

A spectrogram is a visual way of representing the signal strength of a signal over time at various frequencies present in a particular waveform. Mel spectrogram is a spectrogram that is converted to a Mel scale. The Mel scale mimics how the human ear works, with research showing humans don’t perceive frequencies on a linear scale, rather perceive frequencies on a logarithmic scale. Humans are better at detecting differences at lower frequencies than at higher frequencies. 

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

![image6](https://user-images.githubusercontent.com/78913275/175576026-16160a7c-3f03-43f4-9c8b-fb54f96de095.png)

![image7](https://user-images.githubusercontent.com/78913275/175560546-7c89dd72-8709-4ae0-b8ec-8561f43efa85.png)

On the test set, we have achieved an accuracy of 89% and an accuracy of 74.76% on the validation set. 
We still plan on improving the accuracy of our model. 

## A snippet of the predicted output is shown below 

![image8](https://user-images.githubusercontent.com/78913275/175576199-252e8deb-35fc-4f57-8c75-0a72ccc36d9f.png)

## Conclusion 

Through this project, we learnt and showed how we can extract audio features using Librosa and implement deep learning using Keras to predict emotion from speech audio data and some insights on the human expression of emotion through voice. 

## References
1. https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0 

2. https://www.youtube.com/watch?v=gZmobeGL0Yg&list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU 

3. https://www.youtube.com/watch?v=tDaGT4N4aCA&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL 
