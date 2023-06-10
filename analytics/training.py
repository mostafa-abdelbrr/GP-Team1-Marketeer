
import os
import cv2
import pickle
from datetime import datetime
from sklearn.svm import SVC

from skimage.feature import hog

import numpy as np

import matplotlib.pyplot as plt

plt.ioff()

from PIL import Image

def auto_rotate_image(image_path, angle):
    image = Image.open(image_path)
    rotated_image = image.rotate(angle, expand=True)
    file_name, extension = image_path.split('.')
    new_file_name = f"{file_name}_{angle}.{extension}"
    rotated_image.save(new_file_name)
    return new_file_name


def train(training_path):
    labels = {}
    labels_svm = []
    for filename in os.listdir(training_path):
        imageFile = os.path.join(training_path, filename)
        img = cv2.imread(imageFile)
        filename = filename.split('_')
        label = filename[0].replace('.jpg', '')
        if label not in labels:
            labels[label] = []
        for angle in range(360):
            labels[label].append(auto_rotate_image(imageFile, angle))
            labels_svm.append(label)
        os.remove(imageFile)
    hogFeatures = []
    for label in labels:
        for filename in labels[label]:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img.copy(), (64, 128))
            hogFeatures.append(list(hog(img, 16)))
    # print(f'Training features: {len(hogFeatures)}, {len(labels_svm)}')
    clf = SVC()

    clf.fit(hogFeatures, labels_svm)
    with open(f'HOG-model-internal-dataset.pickle', 'wb') as modelFile:
        pickle.dump(clf, modelFile)
    # print('Saved model file.')

def detectProducts(imgs):
    features = []
    try:
        with open('HOG-model-internal-dataset.pickle', 'rb') as modelFile:
            clf = pickle.load(modelFile)
        for img in imgs:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                pass
            img = cv2.resize(img.copy(), (64, 128))
            # print('here')
            features.append(hog(img, 16))
        # print(f'Image features: {len(features[0])}')
        features = np.array(features)
        predictions = clf.predict(features)
        # print(f'Predictions: {predictions}')
        itemFrequency = {}
        for prediction in predictions:
            if prediction in itemFrequency:
                itemFrequency[prediction] += 1
            else:
                itemFrequency[prediction] = 1
        item_axis = []
        frequency_axis = []
        for item in itemFrequency:
            item_axis.append(item)
            frequency_axis.append(itemFrequency[item])
        plt.figure()
        plt.bar(item_axis, frequency_axis)
        plt.title('Frequncy of products')
        plt.xlabel('product name')
        plt.ylabel('frequency name')
        plt.savefig('analytics//ProductFrequency')
        
    except Exception as e:
        # print(f'Training error: {e}')
        pass
    return 