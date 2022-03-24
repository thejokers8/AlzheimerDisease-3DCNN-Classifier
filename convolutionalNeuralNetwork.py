import os
from scipy import ndimage
from os import path
import itertools
import csv
from numpy import *
import time
import matplotlib.pyplot as plt
import nilearn
from nilearn import plotting
import numpy as np
import nibabel as nb
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
import streamlit as st


# app title
st.title('Alzheimer desease \U0001F9E0 predictor ')

# getting data
# data = np.load('C:/Users/Cristhiandcl8/Downloads/adniDatasetResize.npy')
# diagnosis = np.load('C:/Users/Cristhiandcl8/Downloads/ADNIdiagnosis.npy')
# print('Done loading data')


def main(image):
    # diagnosis = st.text_input(
    #    'write the label')

    data = nb.load(f'C:/Users/camil/Downloads/{image[0].name}').get_fdata()
    # target = int(diagnosis)

    normdata = (data - data.min()) / (data.max() - data.min())


    # resize volume
    def resize_volume(img):
        """Resize across z-axis"""
        # Set the desired depth
        desired_depth = 64
        desired_width = 128
        desired_height = 128
        # Get current depth
        current_depth = img.shape[-1]
        current_width = img.shape[0]
        current_height = img.shape[1]
        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height
        # Resize across z-axis
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
        return img

    data = resize_volume(data)
    # expand dimensions to fill CNN
    data = np.expand_dims(data, axis=-1)
    data = np.expand_dims(data, axis=0)
    # plt.imshow(np.rot90(np.squeeze(data[45, :, :, 32])), 'gray')

    #normdata = resize_volume(normdata)
    # showing data
    print(normdata.shape)
    st.title('Volume \U0001F9E0 loaded')
    st.write(f'C:/Users/camil/Downloads/{image[0].name}')
    #col1, col2, col3 = st. columns(3)
    #col1.header("Transversal")
    #line1= st.markdown("<h2 style='text-align: left; color: green;'>Some title</h2>", unsafe_allow_html=True)
    #line2= st.markdown("<h2 style='text-align: center; color: green;'>Some title</h2>", unsafe_allow_html=True)
    #line3= st.markdown("<h2 style='text-align: right; color: green;'>Some title</h2>", unsafe_allow_html=True)
    #col3.header("Sagital")
    #st.pyplot.text(0, 0, SAGITAL, fontdict=None, )
    st.set_option('deprecation.showPyplotGlobalUse', False)

    ax = plt.subplot(1, 3, 1)

    plt.imshow(np.rot90(normdata[:, :, 193 // 2]), cmap='gray')
    ax.axes.xaxis.set_ticks([])
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Axial')

    ax = plt.subplot(1, 3, 2)
    plt.imshow(np.rot90(normdata[:, 229//2, :]), cmap='gray')
    ax.axes.xaxis.set_ticks([])
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Frontal')
    ax = plt.subplot(1, 3, 3)
    plt.imshow(np.rot90(normdata[193//2, :, :]), cmap='gray')
    ax.axes.xaxis.set_ticks([])
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Sagital')

    #plt.legend([line1, line2,line3], ["line1", "line2",'line3'], bbox_to_anchor=(0.5, 1.0), ncol=3)

    st.pyplot()
    # getting one hot encoding output
    # target = to_categorical(diagnosis, num_classes=3)
    # print(target)
    # print('Done one hot')

    # split dataset into training and test set
    # x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=8)
    # print('Done split')


    # creating model
    def cnn():
        model = Sequential()

        model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='glorot_uniform', input_shape=(128, 128, 64, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(BatchNormalization(center=True, scale=True))
        model.add(Dropout(0.5))

        model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='glorot_uniform')) # he_uniform
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(BatchNormalization(center=True, scale=True))
        model.add(Dropout(0.5))

        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='glorot_uniform'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(BatchNormalization(center=True, scale=True))
        model.add(Dropout(0.5))

        model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='glorot_uniform'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(BatchNormalization(center=True, scale=True))
        model.add(Dropout(0.5))

        # model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
        # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        # model.add(BatchNormalization(center=True, scale=True))
        # model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='glorot_uniform'))
        # model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(3, activation='softmax'))
        model.summary()
        print('Done creating model')

        # compiler
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])


        # Loading trained model
        model.load_weights('C:/Users/camil/Downloads/3D_last_session.h5')
        print('Done loading pretrained model')
        return model


    # predictions
    def prediction():
        model = cnn()
        result = model.predict(data)
        return result


    output = prediction()
    out = ''.join([{'0': 'CN', '1': 'MCI', '2': 'AD'}[i] for i in str(np.argmax(output))])
    # tar = ''.join([{'0': 'CN', '1': 'MCI', '2': 'AD'}[i] for i in str(np.argmax(target))])
    st.title('Volume prediction')
    st.write(f'Subject with {out}')
    # st.write(f'CNN prediction: {out}\nCNN target:{tar}')


image = st.file_uploader(label='Upload MRI here', accept_multiple_files=True)

if st.button("Let's see our system prediction"):
    main(image)
