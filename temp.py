# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from PIL import Image    
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

images = pickle.load(open('filenames3.pkl','rb'))

feature_list = np.array(pickle.load(open('embeddings3.pkl','rb')))


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('/Users/siddharthanand/Downloads/images/1545.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances,indeces = neighbors.kneighbors([normalized_result])



import time
for i in indeces[0]:
    img = mpimg.imread(images[i][43:])
    imgplot = plt.imshow(img)
    plt.show()
    time.sleep(1)

