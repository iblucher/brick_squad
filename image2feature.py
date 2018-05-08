import glob
import numpy as np
from PIL import Image 

# get list of files
filelist = glob.glob('samples/*.jpg')
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
x = x_multidim.reshape(n_samples, -1)
# print(x.shape)
# print(x)
