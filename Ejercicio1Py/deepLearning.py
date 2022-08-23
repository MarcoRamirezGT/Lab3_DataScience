import random
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Defining our inputs and replacing any na's with 0's (a whitespace)
xTrain = train.fillna(0)
xTest = test.fillna(0)

# One hot encoding our labels
yTrain = to_categorical(xTrain['label'])
del xTrain['label']

# Reshape to look like an actual square
xTrain = xTrain.values.reshape(-1, 28, 28, 1)
xTest = xTest.values.reshape(-1, 28, 28, 1)

displayList = []

# Pick out some random image
for x in range(9):
    displayList.append(random.randint(1, len(xTrain)+1))

# Show them off
for x in range(len(displayList)):
    plt.subplot(331 + x)
    plt.imshow(xTrain[displayList[x]][:, :, 0])

model = Sequential()
