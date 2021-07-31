from utils_de import *
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing.pool import Pool
import cv2 as cv

import csv

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dsize = (512,512)

def process(image_list):  
    for image_path in image_list: 
        name = image_path.split('/')[-1]
        dst_image_path = os.path.join('./data/image', name)
        dst_mask_path = os.path.join('./data/mask', name)
        try:
            img = imread(image_path)
            img, mask = preprocess(img)
            img = cv.resize(img, dsize)
            mask = cv.resize(mask, dsize)
            imwrite(dst_image_path, img)
            imwrite(dst_mask_path, mask)
        except:
            print(image_path)
            continue

if __name__=="__main__":
    
    image_list = glob.glob(os.path.join('./data/sample', '*.jpeg'))
                
    patches = 16
    patch_len = int(len(image_list)/patches)
    filesPatchList = []
    for i in range(patches-1):
        fileList = image_list[i*patch_len:(i+1)*patch_len]
        filesPatchList.append(fileList)
    filesPatchList.append(image_list[(patches-1)*patch_len:])

    # mutiple process
    pool = Pool(patches)
    pool.map(process, filesPatchList)
    pool.close()
