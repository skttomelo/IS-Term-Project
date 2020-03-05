import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFilter

import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as image

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

# Nicholas' Watson images
WATSON_PATH = './watson/'

watson = []
for i in os.listdir(WATSON_PATH):
    watson.append(np.array(image.open(f'{WATSON_PATH}/{i}')))


train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

# Nicholas' Watson images
num_watson = len(os.listdir(WATSON_PATH))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

#total prediction?
total_predict = num_watson

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)
print('total watson images:', num_watson)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
print("Total prediction images:", total_predict)

batch_size = 128
epochs = 15
IMG_SIZE = 150 # width and height

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
prediction_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our prediction data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SIZE, IMG_SIZE),
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_SIZE, IMG_SIZE),
                                                              class_mode='binary')

predict_data_gen = prediction_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=WATSON_PATH,
                                                                  shuffle=True,
                                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                                  class_mode='binary')

sample_training_images, _ = next(train_data_gen)
predict_images, _ = next(predict_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# shows us the first 5 images of the training images
# plotImages(sample_training_images[:5])
# plotImages(predict_images[:5])

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

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# pred_model = Sequential([
#     Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
#     MaxPooling2D(),
#     Conv2D(32, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Conv2D(64, 3, padding='same', activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(1)
# ])
#
# pred_model.compile(optimizer='adam',
#                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                    metrics=['accuracy'])
#
# pred_model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# predict_history = pred_model.fit_generator(
#     train_data_gen,
#     steps_per_epoch=total_train // batch_size,
#     epochs=epochs,
#     validation_data=predict_data_gen,
#     validation_steps=total_predict // batch_size
# )
# pred_acc = predict_history.history['accuracy']
# pred_val_acc = predict_history.history['val_accuracy']
# pred_loss=history.history['loss']
# pred_val_loss=history.history['val_loss']
# ^ relates to testing prediction stuff


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, pred_acc, label='Training Accuracy')
# plt.plot(epochs_range, pred_val_acc, label='Prediction Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Prediction Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, pred_loss, label='Prediction Loss')
# plt.plot(epochs_range, pred_val_loss, label='Prediction Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Prediction Loss')

plt.show()

with open('itworks.txt', 'r') as f:
    for line in f.readlines():
        print(line)