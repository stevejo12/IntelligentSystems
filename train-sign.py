import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
import cv2
import tensorflow as tf

#reading files to train
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

#get the labels (expected result) for each row
train_labels = train['label'].values
test_labels = test['label']

#list of unique labels
unique = np.array(train_labels)

#get the features for each row
train.drop('label', axis=1, inplace=True)
test.drop('label', axis=1, inplace=True)

# getting the values of pixel 1-784 for each row
# reshape the array to be 28x28 = 784 for each picture
# flattening the individual image into 1D array instead of 2D
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)

x_train, x_test, y_train, y_test = train_test_split(images, train_labels, test_size = 0.25, random_state = 101)

# scaling rgb values 0-255 to between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# reshape back to 28x28 pixels array
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# CNN Model
batch_size = 128
num_classes = 24
epochs = 100

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 , 1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data = (x_test, y_test),
                        epochs=epochs, batch_size=batch_size)

# reshape test data
test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])

# transform test data
label_binarizer = LabelBinarizer()
test_labels = label_binarizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# prediction based on test features or images pixel
y_pred = model.predict(test_images)

# accuracy of the test compared to the actual result.
print("Predicted Accuracy: {} %".format((accuracy_score(test_labels, y_pred.round())* 100)))

model.save('sign.model')










