import cv2 as cv
import pandas as pd
import numpy as np
import os.path
from datetime import datetime
import gc
import pickle
from art_utils import image_info

def now():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "["+ current_time + "]"

def save(filename, img_object):
    print(now() + ": Start")
    with open(filename, 'wb') as f:
        pickle.dump(img_object, f)
        print(f"Saved Image Object to {filename}!")
    print(now() + ": End")
    return ''
        
def load(filename):
    print(now() + ": Start")
    if (os.path.exists(filename)):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(now() + ": End")
        return data
    else:
        raise FileNotFoundError(f"could not find {filename}")
    
def now():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "["+ current_time + "]"


class Images():
    def __init__(self, df):
        
        self.df = df.copy()
        self.df['file_location'] = self.df['new_filename'].apply(self.__output_file__)

        self.train = self.df[self.df.group == 'train'].copy()
        self.test = self.df[self.df.group == 'test'].copy()
        self.train_X = None
        self.test_X = None
        
        self.train_Y = self.train['style']
        self.test_Y = self.test['style']
        
    def read_images(self, img_type=cv.IMREAD_COLOR):
        print(f"{now()}: Started reading images")

        self.test_X = [image_info(cv.imread(fp, img_type)) for fp in self.test['file_location'].copy()]
        print(f"{now()}: Successfully read {len(self.test_X)} test objects")

        self.train_X = [image_info(cv.imread(fp, img_type)) for fp in self.train['file_location'].copy()]
        print(f"{now()}: Successfully read {len(self.train_X)} train objects")
    
        
#     def read_images(self, img_type=cv.IMREAD_COLOR, resized=False):
#         print(f"{now()}: Started reading images")

#         test_images = [cv.imread(fp, img_type) for fp in self.test['file_location'].copy()]
#         print(f"{now()}: Successfully read {len(test_images)} test objects")
            
#         train_images = [cv.imread(fp, img_type) for fp in self.train['file_location'].copy()]
#         print(f"{now()}: Successfully read {len(train_images)} train objects")

        
#         if resized: 
#             print(f"{now()} resizing test images")
#             self.test_X = [cv.resize(im, (256, 256)) for im in test_images]
#             del test_images
#             print(f"{now()} finished resizing test images")
                  
#             self.train_X = [cv.resize(im, (256,256)) for im in train_images]
#             del train_images
#             print(f"{now()} finished resizing train images")
            
#         else:
#             self.test_X = test_images
#             del test_images
#             self.train_X = train_images
#             del train_images
        
#        gc.collect()
            
                                     
#         print(f"{now()}: Successfully read {len(self.test_X)} test objects")
    
    
#         self.train_X = [cv.imread(fp, img_type) for fp in self.train['file_location'].copy()]
#         print(f"{now()}: Successfully read {len(self.train_X)} train objects")
        
        #/Users/juliecorfman/ArtClassifier/train
        ## change depending on file
    
    def __output_file__(self, filename):
        
        
        
        fp = str('/Users/juliecorfman/ArtClassifier/') + "train/" + filename
        if os.path.exists(fp):
            return fp
        else:
            fp = str('/Users/juliecorfman/ArtClassifier/') + "test/" + filename
            if os.path.exists(fp):
                return fp
            else:
                return f"could not find filepath for {filename}"