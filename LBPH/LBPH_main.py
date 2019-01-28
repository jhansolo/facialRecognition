# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 20:04:28 2019

@author: jh
"""
import LBPH
import numpy as np
import cv2

threshold=1.7                                   #hyperparameter
right=0                                         #total guesses
total=0                                         #number of right guesses
testNum=50                                      #number of random tests to perform

studentNum=20                                   #make sure that this studentNum is set to the same studentNum in LBPH_train.py
subjectRange=range(1,studentNum+1)

#make sure these gridX and gridY are the same as those in LBPH_train.py
gridX=8
gridY=8

readPath='base.csv'
falsePath='false.csv'
truePath='true.csv'
loadPath='cambridge\s{}\{}.pgm'

"""reads the known histograms, false positive and true postiive indicator. see details in LBPH.py, 'closeness' function"""
base=LBPH.readBase(readPath).values
falseMetric=np.loadtxt(falsePath,delimiter=",")
trueMetric=np.loadtxt(truePath,delimiter=",")

falseMax=falseMetric.max()
trueMin=trueMetric.min()



"""tests n number of random faces. tallies the accuracy percentage"""
"""optionally can uncomment the commented lines below to see a pop-up window that shows id progression"""

#durationPerImg=500
for k in range(0,testNum):                           #number of times to test
    
    testId=np.random.randint(1,41)              #random subject
    testFace=np.random.randint(4,11)            #random face of the subject. note that currently set such that there are no overlap with known faces
    testPath=loadPath.format(testId,testFace)

    print("actual student # ",testId)
    img,newImg,xEnd,yEnd=LBPH.load(testPath) 

    for i in range(1,xEnd):
        for j in range(1,yEnd):
            LBPH.binary(i,j,img,newImg)
            
    histoVec=(np.matrix(LBPH.histo(gridX,gridY,newImg,xEnd,yEnd))).T
    distVec=base-histoVec
    dist=np.linalg.norm(distVec,axis=0)
    metric=LBPH.closeness(dist)
#
    if metric<=threshold*trueMin:
        print('guessed not in database')
        guess=-1
#        group=LBPH.collage(subjectRange,loadPath,img,newImg,'new-01.jpg',guess)   
    else:
        ID=np.argsort(dist)[0]
        guess=ID+1
#        group=LBPH.collage(subjectRange,loadPath,img,newImg,'pointer.jpg',guess)   
        print("guessed student # ",guess)
    
    if testId>20 and guess==-1:
        right+=1
    elif testId==guess:
        right+=1
    total+=1
    print("-----------------")

#    cv2.imshow("",group)
#    cv2.waitKey(durationPerImg)
#   
#cv2.destroyAllWindows()
print("summary: ")
print("known students #1 - {}".format(studentNum))
print("total number of test images: ",k+1)
print("accuracy :",right/total*100)
