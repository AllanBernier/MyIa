import os
from sklearn.utils import shuffle
import cv2
import numpy
import tensorflow as tf
import numpy as np
from PIL import Image
from numpy import asarray
from keras.datasets import mnist

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers

#CONSTANTES DU PROJET

class_names = ['NORMAL', 'PNEUMONIA']
class_names_label = {class_names:i for i, class_names in enumerate(class_names)}
nb_classes = len(class_names)
IMAGE_SHAPE = (50,100,1)
IMAGE_SIZE = (100,50)
BATCH_SIZE = 128 
EPOCHS = 12

print(class_names_label)
print(f"nombre de classes : {nb_classes}")
print(f"Image size : {IMAGE_SHAPE}")

# ON RECUP LES DONNEES ET ON MODIFIE LEURS SHAPE

def load_data(category):
    DIRECTORY = r"C:/Users/Allan/Documents/Epitech/T8 - Zoidberg/chest_Xray"
    
    
    path = os.path.join(DIRECTORY, category)
    images = []
    labels = []
    
    print(f'loading data from [{category}]')

    for folder in os.listdir(path):
        label = class_names_label[folder]

        for file in os.listdir( os.path.join(path, folder) ):
            img_path = os.path.join( path, folder, file )

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMAGE_SIZE)
        
            images.append(img)
            labels.append(label)
    images = np.array(images, dtype = 'float32') 
    labels = np.array(labels, dtype = 'int32') 
    
    print(f'len images {len(images)}, len labels {len(labels)}')
    return (images, labels)              

(train_input, train_expected) = load_data("train")
(test_input, test_expected) = load_data("test")

train_input, train_expected = shuffle(train_input, train_expected, random_state=25)

# # convert class vectors to binary class matrices
# train_expected = keras.utils.to_categorical(train_expected, nb_classes)
# test_expected = keras.utils.to_categorical(test_expected, nb_classes)

print(train_expected[2])

model = keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, input_shape = IMAGE_SHAPE),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(nb_classes, activation=tf.nn.softmax),      
    ])

model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_input, train_expected, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
score = model.evaluate(test_input, test_expected, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])