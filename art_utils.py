import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from skimage.filters.rank import entropy
from skimage.morphology import disk

x = np.random.random(12)

# for local maxima
argrelextrema(x, np.greater)

def convert_color(img): # to BGR --> RGB
    """ 
    CONVERTS BGR -> RGB
    """
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def mean_RGB(img, l=8,show=False):
    """
    Gets the mean RGB values for an image (cv.show)
    Normalizes the y-axis to [0,1]; x axis is [0,256] 
    Which shows the darkness
    
    Returns an array of 24 values
    Each interval is length of 32 within [0,256]
    
    Also returns the max(h(i))
    
    [0-8] : B
    [8-16] : G
    [16-24] : R
    """
    
    if 256 % l != 0:
        raise Exception("The number of splits must be divisible by 256")
    
    color = ('b','g','r')
    means = []
    maxes = []
    for i,col in enumerate(color):
        histr = cv.calcHist([img],[i],None,[256],[1,256])
        y = histr.reshape(1,256)[0]
        y_normalized = [float(i) / max(y) for i in y]
        maxes.append(max(y_normalized))
        #plt.plot(y_normalized,color = col)
        #plt.xlim([0,256])
        
        split_rgb = np.array_split(y_normalized,l)
        
        means.extend([np.mean(interval) for interval in split_rgb])
        
        if show:
            plt.plot(y_normalized,color = col, label=col)
            plt.xlim([0,256])
            
        
    key = [c.title() + str(n) for c in color for n in range(1,l+1)]
    
    if show:
        plt.legend()
        plt.show()
            
    return means, maxes,key


def mean_HSV(img, l=8, show=False):
    """
    img: image input
    """
    img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = img2[:,:,0], img2[:,:,1], img2[:,:,2]
    means = []
    local_maxima = []
    display = {'h':'b','s':'g','v':'r'}
    
    if 256 % l != 0:
        raise Exception("The number of splits must be divisible by 256")
        
    color = ("h", "s", "v")
    
    for item, s in zip((h,s,v),color):
        histr = cv.calcHist([item],[0],None,[256],[0,256])
        y = histr.reshape(1,256)[0]
        y_normalized = [float(i) / max(y) for i in y]
        
        #plt.plot(y_normalized, color='r', label=s)
        
        split_hsv = np.array_split(y_normalized,l)
        
        means.extend([np.mean(interval) for interval in split_hsv])
        
        ## LOCAL MAXIMA
        max_ind = argrelextrema(y, np.greater,order=50, mode='wrap')
        local_maxima.append(len(max_ind[0]))
        
        if show:
            plt.plot(y_normalized, color=display[s], label=s)
            
        
    
    key = [c.title() + str(n) for c in color for n in range(1,l+1)]
    
    if show:
        plt.legend()
        plt.show()
    
    return means, key, local_maxima

## Lumnosity

def get_splits(y,x):
    """
    th: top half
    bh: bottom half
    lh: left half
    rh: right half
    """
    th = y // 2
    rh = x // 2
    bh = th
    lh = rh
    
    
    if (y % 2 == 1):
        th += 1
        
    if (x % 2 == 1):
        rh += 1
    
    return th,bh,rh,lh

def quadrant_avg(m):
    """
    Get the mean value per quadrant 
    """
    th,bh,rh,lh= get_splits(m.shape[0], m.shape[1])
    # q1
    q1 = m[:th].T[lh:].T
    # q2
    q2 = m[:th].T[:rh].T
    # q3
    q3 = m[bh:].T[:rh].T
    # q4
    q4 = m[bh:].T[lh:].T
    
    lum = [np.mean(x) for x in [q1,q2,q3,q4]]
    return lum

def lumnosity(img):
    """
    img: image input
    returns the mean lumnosity value per quadrant
    """
    y,x,d = img.shape
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    
    whole = [c.reshape(y*x,) for c in [b,g,r]]
    
    lum_vector = ((0.2126*whole[2]) + (0.7152*whole[1]) + (0.0722*whole[1])) / 255.0
    
    lum_percentages_reshaped = lum_vector.reshape((y,x))
    mean_lum = quadrant_avg(lum_percentages_reshaped)

    return mean_lum

## edge ratio
def edge_ratio(img, show=False):
    edges = cv.Canny(img,150,300)
    if show:
        plt.imshow(convert_color(edges))
    prod = (edges.shape[0] * edges.shape[1])
    num_edges = sum([1 for x in edges.reshape(1, edges.shape[0] * edges.shape[1])[0] if x > 0])

    
    ratio_edges = num_edges / prod
    return ratio_edges

def edges(img, show=False):
    edges = []
    
    if show:
        plt.subplot(1,3,1)
        plt.title('Normal')
        
    n = edge_ratio(img, show)
    
    if show:
        plt.subplot(1,3,2)
        plt.title('Blurred')
    blur = cv.GaussianBlur(img,(5,5),0)
    b  = edge_ratio(blur, show)
    
    unsharp_masking =(-1/256) * np.array(
    [[1, 4, 6, 4, 1],
    [4,16,-24,16,4],
    [6,24,-476,24,6],
    [4,16,-24,16,4],
     [1, 4, 6, 4, 1]])
    
    if show:
        plt.subplot(1,3,3)
        plt.title('Sharpen')

    sharpen = cv.filter2D(img, -1, unsharp_masking)
    s = edge_ratio(sharpen, show)
    
    return n,b,s

def dark_ratio(img):
    return sum(np.where(img.reshape(1,np.product(img.shape))[0] <= 64, 1, 0)) / np.product(img.shape)

def entropy_quadrant(img, show=False):
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    entropy_img = entropy(image, disk(5))
    if show:
        img1 = plt.imshow(entropy_img, cmap='gray')
    return quadrant_avg(entropy_img)