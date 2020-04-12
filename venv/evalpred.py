import os
import cv2
import pickle
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# model = tf.keras.models.load_model('cnn-32-batch-no-aug.model')
model = tf.keras.models.load_model('cnn.model')
test_x = pickle.load(open('test_x.pickle', 'rb'))
test_x = test_x/255.0
test_y = pickle.load(open('test_y.pickle', 'rb'))
categories = ('Dog', 'Cat')
DIRECTORY = './Test'
IMG_SIZE = 150

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def load_images(CATEGORIES, DATADIR, IMG_SIZE):
    images = []
    test_images = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for imgs in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, imgs))
                new = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(new)
                test_images.append(new.reshape(-1, IMG_SIZE, IMG_SIZE, 3))
            except:
                pass
    # random.shuffle(images)
    return test_images, images


(test_images, images) = load_images(categories, DIRECTORY, IMG_SIZE)
predictions = []
correct = 0
# evaluation = model.evaluate(test_x, test_y, batch_size=32)
for (i, img) in enumerate(test_images):
    predictions.append(model.predict_classes([img]))
    if predictions[i][0][0] == test_y[i]:
        correct += 1
    print(f'Prediction={categories[predictions[i][0][0]]}, Actual={categories[test_y[i]]}')
    # print(f'{predictions[i]}, real ans: {test_y[i]}')
print(f'Percentage_Correct={correct/len(predictions)}')

# plotImages(images[:10])
# print(predictions)