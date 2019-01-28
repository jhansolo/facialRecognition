# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:46:09 2018

@author: jh
"""

import numpy as np
import cv2

dir_path=r"C:\Users\jh\eigenFace\cambridge\s41\{}.jpg"              ##location of training data
for i in range(1,11):
    firstPath=dir_path.format(i)
    firstPic=cv2.imread(firstPath,0)
    cv2.imwrite("{}.pgm".format(i),firstPic)


