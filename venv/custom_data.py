import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle # for saving data
import numpy as np
from PIL import Image

epochs = 15
IMG_SIZE = 150 # width and height

DATADIR = './PetImages'
CATEGORIES = ('Dog', 'Cat')

training_data = []
def create_training_data_PIL():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for imgs in os.listdir(path):
            try:
                img = Image.open(os.path.join(path, imgs))
                new = img.resize((IMG_SIZE,IMG_SIZE))
                training_data.append([np.array(new),class_num])
            except:
                pass
    random.shuffle(training_data)


def create_training_data_cv2():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for imgs in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, imgs))
                new = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                training_data.append([new,class_num])
            except:
                pass
    random.shuffle(training_data)


create_training_data_cv2()

train_x = [] # Features
train_y = [] # Labels

for features, label in training_data:
    train_x.append(features)
    train_y.append(label)
print(train_x)
train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3) # 3 means that we are using color, you would use 1 if it was gray scale
train_y = np.array(train_y)


with open('train_x.pickle', 'wb') as f:
    pickle.dump(train_x, f)

with open('train_y.pickle', 'wb') as f:
    pickle.dump(train_y, f)