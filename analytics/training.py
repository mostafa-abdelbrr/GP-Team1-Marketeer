
import os
import cv2
import pickle
from datetime import datetime
from sklearn.svm import SVC

from skimage.feature import hog

import numpy as np

import matplotlib.pyplot as plt

plt.ioff()

def train(training_path):

    labels = {}

    for filename in os.listdir(training_path):
        imageFile = os.path.join(training_path, filename)
        img = cv2.imread(imageFile)
        filename = filename.split('_')
        labels[filename[0].replace('.jpg', '')] = imageFile

    hogFeatures = []

    for label in labels:
        img = cv2.imread(labels[label], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img.copy(), (128, 256))
        hogFeatures.append(list(hog(img, 16)))

    clf = SVC()
    clf.fit(hogFeatures, list(labels.keys()))
    with open(f'HOG-model-internal-dataset.pickle', 'wb') as modelFile:
        pickle.dump(clf, modelFile)

def detectProducts(imgs):
    features = []
    # with open('HOG-model-internal-dataset-2023-06-03 21-22-09.702882.pickle', 'rb') as modelFile:
    #     clf = pickle.load(modelFile)
    try:
        with open('HOG-model-internal-dataset.pickle', 'rb') as modelFile:
            clf = pickle.load(modelFile)
        # print(f'Length of imgs is: {len(imgs)}')
        for img in imgs:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                pass
        # print(img.shape)
            img = cv2.resize(img.copy(), (128, 256))
            features.append(list(hog(img, 16)))
        # clf.predict(features)
        features = list(features)
        predictions = clf.predict(features)
        histogram = np.histogram(predictions, bins=len(np.unique(predictions)))
        plt.figure()
        plt.plot(histogram)
        plt.savefig('analytics/histogram.png')
    except Exception as e:
        # print(f'Training error: {e}')
        pass
    return 