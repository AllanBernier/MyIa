import os
from random import shuffle
import cv2
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers

#CONSTANTES DU PROJET

class_names = ['zero','un','deux','trois','quatre','cinq','six','sept','huit','neuf']
class_names_label = {class_names:i for i, class_names in enumerate(class_names)}
nb_classes = len(class_names)
IMAGE_SHAPE = (28,28,1)
BATCH_SIZE = 128
EPOCHS = 15

print(class_names_label)
print(f"nombre de classes : {nb_classes}")
print(f"Image size : {IMAGE_SHAPE}")

# ON RECUP LES DONNEES ET ON MODIFIE LEURS SHAPE
(train_input, train_expected), (test_input, test_expected) = mnist.load_data()

print(f'\t SHAPE OF DS IS {train_input.shape}')

train_input = train_input.astype("float32") / 255
test_input = test_input.astype("float32") / 255

train_input = np.expand_dims(train_input, -1)
test_input = np.expand_dims(test_input, -1)

print("x_train shape:", train_input.shape)
print (train_expected[2], class_names[train_expected[2]] )

# convert class vectors to binary class matrices
train_expected = keras.utils.to_categorical(train_expected, nb_classes)
test_expected = keras.utils.to_categorical(test_expected, nb_classes)

print(train_expected[2])

model = keras.Sequential(
    [
        keras.Input(shape=IMAGE_SHAPE),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(nb_classes, activation="softmax"),
    ]
)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train_input, train_expected, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

score = model.evaluate(test_input, test_expected, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])