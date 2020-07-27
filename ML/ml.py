import tensorflow as tf
import numpy as np
import cv2
import os
import random
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Path of files on local device
data = "D:\\innovation cell\\data"
img_size = 50

#I separated test data
test_data_path = os.path.join(data, 'test')

test_data=[]
for t in os.listdir(test_data_path):
    #labelling the data
    if t[0] == 'N':
        class_num =0
    elif t[1] == 'Y':
        class_num = 1
    else:
        continue
    
    img = cv2.imread(os.path.join(test_data_path, t),0)
    img = cv2.resize(img, dsize=(img_size, img_size))
    test_data.append([img, class_num])
    
#shuffling data    
random.shuffle(test_data)
x_test=[]
y_test=[]
for f, l in test_data:
    x_test.append(f)
    y_test.append(l)

#converting data into appt shaped numpy arrays, normalizing data so that model can learn easily
x_test = np.array(x_test).reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)
x_test=x_test/255.0


#after experimenting with many values, this seemed like a good bet
BatchSize = 80
epochs = 10
steps_per_epoch = BatchSize//epochs

#getting the training data
training_data=[]
for d in os.listdir(data):
    if d[0] == 'N':
        class_num = 0
    elif d[0] == 'Y':
        class_num = 1
    else:
        continue
    img= cv2.imread(os.path.join(data, d), 0)    
    res = cv2.resize(img, dsize = (img_size, img_size))
    training_data.append([res, class_num])

random.shuffle(training_data)
x = []
y = []
for features, labels in training_data:
    x.append(features)
    y.append(labels)
x = np.array(x).reshape(-1, img_size, img_size, 1)


x=x/255.0

#randomly setting first 20 images as the validation set
x_train = x[20:]
x_val = x[:20]
y = np.array(y)
y_train = y[20:]
y_val = y[:20]


# construct the training image generator for data augmentation (since we had less data to work on, I thought this was a good idea)
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")


#creating the structure of the sequential cnn
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape= (img_size, img_size, 1)))#x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
#model.add(Dropout(0.1))  (adding dropout caused evaluated accuracy to dip most of the times so, I added L2 regularizer instead)

model.add(Dense(128, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))


model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

#using 10% of data for validation
model.fit(x, y, batch_size = BatchSize, validation_split = 0.05, epochs=epochs)


#Data augmentation didn't give good enough results

#augmented_data = aug.flow(x_train, y_train, batch_size=BatchSize)
#H = model.fit(augmented_data , validation_data=(x_val, y_val), steps_per_epoch=steps_per_epoch, epochs=epochs)




print("Evaluating on validation set: ")
results  = model.evaluate(x_test, y_test, batch_size= BatchSize)
print("test_loss, test_accuracy: ", results[0], results[1])
