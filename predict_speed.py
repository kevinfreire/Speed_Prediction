import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet152
import numpy as np
from numpy import asarray
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def get_data(path, input_shape, limit):
    
    x_data = []
    y_data = []
    i=0
    # Iterate through the names of contents of the folder
    for img in os.listdir(path):
        img = cv.imread(path+img)
        x_data.append(np.resize(img, input_shape))
        i += 1
        if(i==limit):
            break

    # Splitting x_data to smaller dataset due to lack of memory
    x_data = np.asarray(x_data)
    x_reduce = x_data[0:100]

    i=0
    with open("target.txt") as target:
        for line in target:
            y_data.append(tf.strings.to_number(line.rstrip()))
            i += 1
            if i==limit:
                break

    y_data =np.asarray(y_data)
    # Slicing y_data to smaller dataset due to lack of memory
    y_reduce = y_data[0:100]

    # spliting data between train (60%) and test (40%) sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.4, shuffle=False)

    return (x_train, x_test, y_train, y_test)

##################################################### MODEL ##############################################################

# Retrieving Data
limit = 200
input_shape = (224, 224, 3)
path = "./train_images/"
(x_train, x_test, y_train, y_test) = get_data(path, input_shape, limit)

# We build the base model
base_model = ResNet152(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))

# We freeze every layer in our base model so that they do not train, we want that our feature extractor stays as before --> transfer learning
for layer in base_model.layers: 
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen.')

# We take the last layer of our the model and add it to our classifier
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(1000, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(1, activation='linear', name='predictions')(x)
model = Model(base_model.input, x)

# We compile the model
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['accuracy'])

# Retrieve Model summary
model.summary()

# We start the training
epochs = 10
batch_size = 10

# We train it
model.fit(x=x_train, y=y_train,
          batch_size=batch_size,
          validation_data=(x_test, y_test),
          epochs=epochs)

# We evaluate the accuracy and the loss in the test set
scores = model.evaluate(x_test, y_test, verbose=1)

model.save("predict_speed.model", save_format="h5")
print("finished saving model")
print('Test loss:', scores[0])

######################################## Testing for accuracy #########################################

# Load model
model = load_model('predict_speed.model')

# We predict the output of the image test set
scores = model.predict(x_test, verbose=1)

# Calculating accuracy with +/- 2 difference
count=0
for i in range(len(y_test)):
    if(y_test[i]-2 < scores[i] and scores[i] < y_test[i]+2):
        count += 1

accuracy = (count / len(y_test)) * 100
print("Accuracy: " + accuracy + "%")
exit()