import glob
import os
import numpy as np
from PIL import Image
from read_label import *
from skmultilearn.adapt import MLkNN
from sklearn.metrics import accuracy_score

dir = "/home/jmi/KU/Block 4/Large Scale Data Analysis/competitions/fashion/"

def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])

def imageprep(path):
    # get list of files
    filelist = glob.glob(path)
    filelist.sort(key=sortKeyFunc)
    filelist = filelist[:1000]
    # filelist = filelist[:10]
    n_samples = len(filelist)
    # print(filelist)
    # print(n_samples)

    # resize images
    images_resized = [Image.open(fname).resize((64,64), Image.ANTIALIAS) for fname in filelist]
    # print(images_resized)

    # convert images to numpy array
    x_multidim = np.array([np.array(image) for image in images_resized])
    #print(x_multidim.shape)

    # flatten the numpy array
    return x_multidim.reshape(n_samples, -1)
    # print(x.shape)
    # print(x)

Xtrain = imageprep(dir + 'tmp_images/*.jpg')

Xval  = imageprep(dir + 'val_images/*.jpg')

i, ytrain = multi_label(dir + "train_subset.json")
i, yval = multi_label(dir + "validation.json")

ytrain, yval = ytrain[:1000], yval[:1000]

classifier = MLkNN(k=10)

classifier.fit(Xtrain, ytrain)

predictions = classifier.predict(Xval)
print(predictions)

# acc = accuracy_score(yval, predictions)
# print("Accuracy on test set: {}".format(acc))
