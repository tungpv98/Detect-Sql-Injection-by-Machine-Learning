#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:04:48 2017

@author: Scott
@problem: recognize cat or dog image.
"""
# Sequential - For initialize NN
# Convolution2D - For add Convolution layer to handle 2D images
# MaxPooling2D - For proceed pooling step(pooling layer)
# Flatten - For proceed flatten to becoming the input of fully connected layer
# Dense - For add classic ANN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# =========== Part 1 - Building CNN model ===========
# Initialising layers for CNN
model = Sequential()

# Step 1 - Convolution
# Generate feature map for given image using feature detector
filters = 32
img_width = 150
img_height = 150
img_dim = 3
feature_detector = (3,3)
model.add(Conv2D(filters,
                 feature_detector,
                 input_shape=(img_width, img_height, img_dim),
                 activation="relu"))

# Step 2 - Pooling
# For reducing the size of feature maps
model.add(MaxPooling2D(pool_size=(2,2)))

# Adding more convolution layer and pooling layer for improve model :D
model.add(Conv2D(filters,
                 feature_detector,
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters,
                 feature_detector,
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


# Step 3 - Flatten
# For get one huge dimensional vector to use input layer
model.add(Flatten())

# Step 4 - Add classic ANN with dropout(regularzation)
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.5)) # I think drop out rate is depend on count of neuron.
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

# Step 5 - Compiling CNN, I think adam or rmsprop is good choice.
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# =========== Part 2 - Fitting CNN model to images ===========
# Apply image argumentation(preprocessing) to avoid overfitting
# because this preprocessing make a lots of new pattern images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("dataset/training_set",
                                                 target_size = (img_width, img_height),
                                                 batch_size = 32,
                                                 class_mode = "binary")

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (img_width, img_height),
                                            batch_size = 32,
                                            class_mode = 'binary')

# steps_per_epoch - number of training set data
# validation_data - test data
# validation_steps - number of test set data
model.fit_generator(training_set,
                    steps_per_epoch = 8000/32,
                    epochs = 100,
                    validation_data = test_set,
                    validation_steps = 2000/32)

# =========== Part 2 - Predict images ===========
from keras.preprocessing import image
import numpy as np
predict_img = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg",
                             target_size=(img_width, img_height))
X_predict = image.img_to_array(predict_img)
X_predict = np.expand_dims(X_predict, axis=0)

result = model.predict(X_predict)
# Check result dog or cat
#training_set.class_indices
if result [0][0] == 1:
    print("Dog")
else:
    print("Cat")
