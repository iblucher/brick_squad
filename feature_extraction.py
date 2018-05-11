import glob
import numpy as np
from PIL import Image
from skimage.feature import greycomatrix, greycoprops

dir = 'C:/Users/Christian/Desktop/images/*.jpg'
filelist = glob.glob(dir)
n_samples = len(filelist)

images_resized = [Image.open(fname).resize((64,64), Image.ANTIALIAS) for fname in filelist]



x_multidim = np.array([np.array(image.convert('L')) for image in images_resized])

x_glcm = np.array([greycomatrix(image, [5], [0], 256, symmetric=True, normed=True) for image in x_multidim])


## HELPER FUNCTIONS ##

def _sumKminus(glcm, k):
    size_i, size_j = glcm.shape[0:2]
    glcm_sum = 0

    for i in range(size_i):
        for j in range(size_j):
            if (abs(i - j) == k):
                glcm_sum += glcm[i, j]

    return glcm_sum

def _sumKplus(glcm, k):
    size_i, size_j = glcm.shape[0:2]
    glcm_sum = 0

    for i in range(size_i):
        for j in range(size_j):
            if (i + j == k):
                glcm_sum += glcm[i, j]

    return glcm_sum
  
def _sumAverageSingle(glcm):
    size_i, size_j = glcm.shape[0:2]
    image_sum = 0
    
    for i in range(2, (size_i * 2) + 1):
        image_sum += i * _sumKplus(glcm, i - 2)

    return image_sum

## FEATURES ##

def _angSecMomement(glcm):

    i_sum = 0
    image_sum = 0

    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            i_sum += glcm[i, j]

        image_sum += i_sum**2
        i_sum = 0

    return image_sum[0][0]

def _correlation(glcm):
    return greycoprops(glcm, 'correlation')[0][0]

def _contrast(glcm):
    return greycoprops(glcm, 'contrast')[0][0]

def _variance(glcm):
    i_sum= 0
    j_sum = 0

    mean = np.mean(glcm)
    
    for i in range(glcm.shape[0]):
        j_sum = 0
        for j in range(glcm.shape[1]):
            j_sum += ((i - mean)**2) * glcm[i, j]

        i_sum += j_sum

    return i_sum[0][0]

def _inverseDiffMoment(glcm):
    i_sum= 0
    j_sum = 0

    for i in range(glcm.shape[0]):
        j_sum = 0
        for j in range(glcm.shape[1]):
            j_sum += ((1.0 / (1.0 + ((i - j)**2.0))) * glcm[i, j])

        i_sum += j_sum

    return i_sum[0][0]

def _sumAverage(glcm):

    image_sum = 0
    
    for i in range(2, (glcm.shape[0] * 2) + 1):
        image_sum += i * _sumKplus(glcm, i - 2)

    return image_sum[0][0]

def _sumVariance(glcm):

    image_sum = 0
    glcm_avg = _sumAverageSingle(glcm)
    
    for i in range(2, (glcm.shape[0] * 2) + 1):
        image_sum += ((i - glcm_avg)**2 * _sumKplus(glcm, i - 2))

    return image_sum[0][0]

def _sumEntropy(glcm):

    image_sum = 0
    glcm += 1

    for i in range(0, (glcm.shape[0] * 2) - 1):
        var = _sumKplus(glcm, i)
        image_sum += var * np.log(var)

    return image_sum[0][0]

def _entropy(glcm):
    glcm += 1
    res = -1.0 * np.sum(np.multiply(glcm, np.log(glcm)))
    return res

def _diffVariance(glcm): # NOT WORKING
    varList = []

    for i in range(0, glcm.shape[0] - 1):
        varList.append(_sumKminus(glcm, i))
    res = np.var(varList)

    return res

def _diffEntropy(glcm): # NOT WORKING

    i_sum = 0

    glcm += 1

    for i in range(glcm.shape[0]):
        sum_k_minus = _sumKminus(glcm, i)

        if(sum_k_minus != 0):
            i_sum += (sum_k_minus * np.log(sum_k_minus))

    res = i_sum * (-1.0)

    return res[0][0]

print(_angSecMomement(x_glcm[0]))
print(_correlation(x_glcm[0]))    
print(_contrast(x_glcm[0]))  
print(_variance(x_glcm[0]))  
print(_inverseDiffMoment(x_glcm[0]))
print(_sumAverage(x_glcm[0]))
print(_sumVariance(x_glcm[0]))
print(_sumEntropy(x_glcm[0]))
print(_entropy(x_glcm[0]))
print(_diffVariance(x_glcm[0]))
print(_diffEntropy(x_glcm[0]))
