# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 20:04:28 2019

establishes a collection of characteristic histograms for the known subjects

trains the dataset to set a metric for determining when a test face belongs 
to one of the known subjects or is a completely new face

@author: jh
"""

import LBPH
import numpy as np


loadPath='cambridge\s{}\{}.pgm'
studentNum=20                           #number of known subjects
studentIds=range(1,studentNum+1)
faceNum=4                               #number of faces per subject
photoIds=range(1,faceNum+1)
gridX=8                                 #number of grids to divide an image into horizontally
gridY=8                                 #number of grids to divide an image into vertically

writePath='base.csv'                    #write the known subjects' histograms here
LBPH.baseLine(loadPath,studentIds,photoIds,gridX,gridY,writePath,faceNum) #see details in LBPH.py

base=LBPH.readBase(writePath).values    #reads the known histograms from before

testId=range(studentNum+1,42)                     #these are the faces not in the known subjects' collection
                                        #use these to train for false-positive
testFace=photoIds                       #same number of faces per test subject as the known subjects

"""writes the false positive metric. see details in LBPH, 'closeness' function"""
falsePath='false.csv'
falseMetric=LBPH.train(base,testId,testFace,loadPath,gridX,gridY)
np.savetxt(falsePath,falseMetric)

"""writes the true positive metric. see details in LBPH, 'closeness' function"""
truePath='true.csv'
trueMetric=LBPH.train(base,studentIds,photoIds,loadPath,gridX,gridY)
np.savetxt(truePath,trueMetric)




