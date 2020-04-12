import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

import os
import pickle
import time
import matplotlib.pyplot as plt

batch_size = 32

NAME = f'Cat_Dog_CNN_{batch_size}_batch_{int(time.time())}'
# tensorboard = TensorBoard(log_dir=f'logs\\{NAME}')
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


train_x = pickle.load(open('train_x.pickle', 'rb'))
train_x = train_x/255.0
train_y = pickle.load(open('train_y.pickle', 'rb'))
IMG_SIZE = 150
epochs = 10
epoch_steps = len(train_x) // batch_size
val_split = 0.3
val_steps = (len(train_x)*.3) // batch_size

# 32 batch size
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
# model = Sequential()
#
# model.add(Conv2D(64, (3,3), input_shape=train_x.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
#
# model.add(Dense(64))
# model.add(Activation('relu'))
#
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_split=val_split)

model.save(f'{NAME}.model')

# # 128 batch size
#
# NAME = f'Cat_Dog_CNN_128_batch_{int(time.time())}'
# tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
# # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
# model = Sequential()
#
# model.add(Conv2D(64, (3,3), input_shape=train_x.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
#
# model.add(Dense(64))
# model.add(Activation('relu'))
#
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.fit(train_x, train_y, steps_per_epoch=epoch_steps, validation_steps=val_steps, batch_size=batch_size[1], epochs=epochs, validation_split=val_split, callbacks=[tensorboard])
#
# model.save(f'{NAME}.model')

