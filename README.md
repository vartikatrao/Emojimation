# Emojimation

---
Facial expression recognition 

IEEE Envision Project

year: 2023

---

### Project Guide

- 

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

###  Facial Expression Recognition 2013

The data consists of 48x48 pixel grayscale images of faces. There are a total of 28,709 images. 

### Bar graph representing the number of images of each emotion: 

![image](https://github.com/ktLearner/Emojimation/assets/122672121/6f0d27e7-959d-4638-a035-e747d248266e)



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

 Convolutional Neural Network or CNN is a type of artificial neural network which has capability to analyse images and video. They apply a special type of mathematical operation called _Convolution_. 

Due to large number of paramters, these networks can learn very complex patterns and are _Shift invariant_. Owing to this power of CNN, they are widely used in image classification algorithms.

 Our project involves use of a CNN architecture to classify input image into 7 classes of emotion. The network learns pattern from images training images and uses the knowledge to classify real time human expressions

![image](https://github.com/Bhuvanesh-Singla/Emojimation_Bhuvanesh_Version/assets/125354611/e07d1fd2-0b02-4693-8d04-1921bbfae833)



## Model Architecture

Our CNN model is inspired by VGG-16 which is one of the popular algorithms for image classification. 

We have 4 convolution layers each followed by a _Dropout_ and _MaxPooling_ layer with a stride of 2 and each layer having few sub-layers. 

A kernel size of 3 and _same_ padding was utilised. 

It is followed by a _Flattening layer_ and a _Dense network_ of 3 fully connected layers and at the end an output layer for performing the final classification. 

While compiling the model, Adam optimizer was deployed with an optimum leraning rate to perform the training effectively.

The activation function used in layers is _ReLU_ and at the end _Softmax_ is used in output layer.

Several techniques to prevent overfitting and ease training had been used.
* 1. Dropout Layers: These randomly turns off certain neurons from the model in order to reduce the over dependency of model over any particular node or feature.
* 2. Batch Normalization : It makes training of neural network faster and more stable by performing some rescaling and recentering of layers' inputs and help in decreasing number of epochs to train.
* 3. Early Stopping : Early Stopping is a technique wherein the training of model is haulted forcefully once it starts to overfit.
* 4. Reduce Learning Rate: When a metric stops improving for a longer time, the model is benefitted by reducing the learning rate by a certain factor.
 

## Result 

![image6](https://user-images.githubusercontent.com/78913275/175576026-16160a7c-3f03-43f4-9c8b-fb54f96de095.png)

![image7](https://user-images.githubusercontent.com/78913275/175560546-7c89dd72-8709-4ae0-b8ec-8561f43efa85.png)

On the test set, we have achieved an accuracy of 89% and an accuracy of 74.76% on the validation set. 
We still plan on improving the accuracy of our model. 

## A snippet of the predicted output is shown below 

![image8](https://user-images.githubusercontent.com/78913275/175576199-252e8deb-35fc-4f57-8c75-0a72ccc36d9f.png)

## Conclusion 

Through this project, we learnt and showed how we can extract audio features using Librosa and implement deep learning using Keras to predict emotion from speech audio data and some insights on the human expression of emotion through voice. 

## Implementation
We have implemented our project by making a GUI based interface and a Web app using Streamlit.

### Streamlit

#### Instructions

1) clone the repo

2) switch to webpp branch

3) add the dl models h5 file with the name "FER2013new.h5"

4) create a virtual env

5) install requirements.txt

6) run this command in terminal "streamlit run webapp.py"

#### Webview
![image](https://github.com/ktLearner/Emojimation/assets/122672121/5de2c414-dcd9-45f0-bcf7-72e037918917)
### GUI


## References
1. https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0 

2. https://www.youtube.com/watch?v=gZmobeGL0Yg&list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU 

3. https://www.youtube.com/watch?v=tDaGT4N4aCA&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL 
