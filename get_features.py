from skimage import io, filters, color, transform
from scipy.fftpack import fftn, fftshift
import numpy as np

def toImage(features):
    return features.reshape(3, 32, 32).transpose(1, 2, 0)

def toFeatures(image):
    return image.reshape(-1)

def toComps(image):
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]

def normalize(features):
    norm = np.linalg.norm(features)
    return features / norm

def FFT(channel):
    filtered_image = filters.difference_of_gaussians(channel, 1, 12)
    wimage = filtered_image * filters.window('hann', channel.shape)
    im_f_mag = fftshift(np.abs(fftn(wimage)))
    return np.log(im_f_mag)
    
def SOBEL_FFT(channel):
    image = filters.sobel(channel)
    image = filters.difference_of_gaussians(image, 0, 12)
    image = image * filters.window('hann', (32, 32))
    image = fftshift(np.abs(fftn(image)))
    return np.log(image)

def FFT_SOBEL(channel):
    filtered_image = filters.difference_of_gaussians(channel, 1, 12)
    wimage = filtered_image * filters.window('hann', channel.shape)
    im_f_mag = fftshift(np.abs(fftn(wimage)))
    image = filters.sobel(np.log(im_f_mag))
    return image