# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:59:47 2018

@author: jay han

implements the eigenface algorithm discussed by Turk and Pentland.
press 'run' in your python IDE to see the results.

2 ways to view the results:
    #1(default)- graphic pop-up window showing the 10 known subjects, the chosen test face, and the eigenface reconstruction
         of the test face. If the test face belongs to one of the 10 known subjects, an arrow will identify that 
         subject. If the test face does not belong to the 10 know subjects, a "new face" message will appear. If the 
         test face is not a face at all, a "not a face" message will appear
    #2 - cycles through the entire database and checks for the accuracy of the algorithm's predicitions against all of them.
         Displays the subject number chosen, the guess made, and overall accuracy statistics
         
uses numpy, pandas, and cv2
"""

"""testing parameters"""
numOfTestImages=100      #number of random test images to cycle through
durationPerImg=500      #duration of each test image on the screen, milliseconds
threshold=0.2           #hyper parameter: lower value for more accuracy among known subjects
                        #                 higher value more false positives

import numpy as np
import pandas as pd
import cv2

def loadFaces(studentRange,faceRange):
## loads all the faces necessary for creating the eigenface subspace stores them 
## all within a pandas DataFrame 

    faceFrame=pd.DataFrame()
    w=0
    for i in studentRange:                                                      ##specifies which student to load            
        for j in faceRange:                                                     ##specifies how many face from each student to use in eigenface construction.  
            loadPath=dir_path.format(i,j)
            img=cv2.imread(loadPath,0)
            imgVec=img.flatten('F')                                             ##originally faces read as a 112x92 matrix. here flattens to a 10304x1 vector
            faceFrame.insert(loc=w,value=imgVec,column="student {}_{}".format(i,j))
            w+=1
        
    return faceFrame

def wrap(vec):
## converts the 10304x1 vector representation of an image back to the 112x92 matrix form of type uint8 for viewing via cv2
    vec=np.split(vec,picWidth)
    vec=np.vstack(vec).T
    vec=np.array(vec,dtype=np.uint8)
    return vec

def pseudoCovariance(frame):
## calculates the MxM pseudo covariance matrix of the M faces that will be used to build the eigenfaces
    faceMatrix=faceFrame.values                                                 ## extracts the values from the DataFrame with all the columns being a 10304x1 face vector     
    averageVec=np.array(faceFrame.sum(axis=1)/frame.shape[1])                   ## average face vectors
    a=faceMatrix.T-averageVec                                                   ## subtract this average face vector from all the face vectors
    a=a.T
    a=np.matrix(a)
    pseudoCov=(a.T)*a                                                           ## calculated pseudo covaraince matrix
    return a, pseudoCov, averageVec

def normalize(original):
## to normalize the columns of an eigenvector matrix     
    reduced=original/np.linalg.norm(original)                                          
    return reduced

def eigen(a,pseudoCov):
## to calculate the eigenvalues and eigenvectors from the pseudo-covariance matrix
    eigValues, pseudoEigVecs=np.linalg.eig(pseudoCov)                           ## eigenvectors (pseudo) calculated here correspond to those of the pseudo-covariance matrix. for eigenvectors correspnding to the real covairance matrix, see below
    sortKey=np.flip(np.argsort(eigValues),0)                                    ## key to sort by largest eigenvalues 
    ##sorted
    eigValues=eigValues[sortKey]                                                ## sort the eigenvalues by the previous key. eigenvectors associated with large eigenvalues contribute more
    sortedPseudoEigVecs=np.matrix(pseudoEigVecs[:,sortKey])                     ## sort the pseudo eigenvectors by same key
    """found eigenValues of the real covariance matrix"""
    realEigVecs=a*sortedPseudoEigVecs                                           ## obtain real eigenvectors associated with the covariance matrix through matrix multiplication with a. see Pentland paper for details
    realEigVecs=np.apply_along_axis(normalize,0,realEigVecs)
    return eigValues, realEigVecs

def loadTest(testPath):
    testFace=cv2.imread(testPath,0)
    testVec=testFace.flatten('F')
    return testVec


def baselineCoords(faceFrame,eigVals,eigFaces,studentRange,faceRange):
## Calculates the unique 30x1 weight vector corresponding to each 10 known subjects
## since 3 images from each of the 10 known subjects are used in building the eigenface subspace, this unique
## 30x1 weight vector will be an average of these 3 image's own respective weight vectors
    weightFrame=pd.DataFrame()                                                  ## empty weight frame. each column will correspond to a Mx1 vector, where M is the number of faces used to build the eigenface subspace
    k=0
    for i in studentRange:
        totalWeight=np.zeros([len(eigVals),1])
        for j in faceRange:
            currentFace=np.matrix(faceFrame["student {}_{}".format(i,j)].values-averageVec).T
            currentWeight=eigFaces.T*currentFace                                ## calculates the weight vector for the current image 
            totalWeight+=currentWeight                                          ## add to the total weight vector    
        totalWeight/=len(faceRange)                                             ## takes average
        weightFrame.insert(loc=k,value=totalWeight.flatten("F"),column="student {}".format(k+1)) ## stores in the weight frame
        k+=1 

    w=weightFrame.values
    return weightFrame, w

def projectFace(testPath,averageVec,eigFaces,baseWeights):
## projects each face onto the eigenface subspace to obtain the unique weight coordinates
## essentailly a multiplication betweent the matrix made of eigenface vectors and the test image vector
## also calculates the distance to each of the known faces
    
    testFace=cv2.imread(testPath,0)
#    cv2.imshow("test face",testFace)

    testVec_original=testFace.flatten("F").T                        
    testVec=np.matrix((testVec_original-averageVec)).T
    testWeight=(eigFaces.T*testVec)                                             ## the weight vector
    testWeight[-1]=0                                                            ## discard last item of the vector, since it contains only information regarding noise
    projectedFace=eigFaces*testWeight
    projectedFace=np.array(projectedFace).flatten("F")+averageVec   
    
    spaceVec=testVec_original-projectedFace                                     ## the difference vector between the test image and its projection in the eigenface subspace
    spaceDist=np.linalg.norm(spaceVec)                                          ## the norm of the previous is the distance between the test face and the overall eigenface subspace. if this is large, then test unlikely to be a face at all
#    cv2.imshow("reconstructed face",wrap(projectedFace))
    diffVec=baseWeights-testWeight
    distVec=np.linalg.norm(diffVec,axis=0)                                      ## the distances to each of the 10 known faces
    
    return testVec_original, projectedFace, spaceDist, distVec

def train(trainStudent,trainRange,dir_path,averageVec,eigFaces,baseWeights):
## training process:
## of the 40 students in the databse, 10 are chosen to be known subjects. The first 3 images of each of these students are used in constructing the eigenface subspace. 
## Then images #4, #5, #6, and #7 of each of these 10 students are used in this training process
## for a single known subject, call it subject A, its images #4, #5, #6, #7 will each have a unique 30x1 weight vector when projected onto the eigenface subspace
## the distnace between the average of these 4 weight vectors and the baseline weight vectors is the average distance from which
## a new image's weight vector is compared. 
    radiusFrame=pd.DataFrame()
    maxSpaceDist=0
    for i in trainStudent:
        average=0
        for j in trainRange:
            trainPath=dir_path.format(i,j)
            testVec,projectedFace,spaceDist,distVec=projectFace(trainPath,averageVec,eigFaces,baseWeights)
            average+=distVec[i-1]
            if spaceDist>=maxSpaceDist:
                maxSpaceDist=spaceDist
        average/=len(trainRange)
        radiusFrame.insert(loc=i-1,value=[average],column="student {}".format(i))
    return radiusFrame,maxSpaceDist

def collage(dir_path, studentRange, testImg,restructImg,message,initPos):
## a visual stitching of the 10 known faces, the test face, its projection, and messages for identification, new face, and not a face
    firstPath=dir_path.format(1,1)
    firstPic=cv2.imread(firstPath,0)
    for i in range(2,11):
        if i==1:
            break
        loadPath=dir_path.format(i,1)
        img=cv2.imread(loadPath,0)
        firstPic=np.concatenate((firstPic,img),axis=1)

    band=np.zeros(firstPic.shape)
    bottom=np.zeros(firstPic.shape)
    for i in range(368,460):
        bottom[:,i]=testImg[:,abs(368-i)]
    for j in range(460, 460+92):
        bottom[:,j]=wrap(restructImg)[:,abs(460-j)]
    for k in range(initPos,initPos+92):
        band[:,k]=message[:,abs(initPos-k)]
    bigPic=np.array(np.concatenate((firstPic,band,bottom),axis=0),dtype=np.uint8)
    cv2.imshow("",bigPic)
    
    
picHeight=112
picWidth=92

"""load training data"""
dir_path=r"cambridge/s{}/{}.pgm"              ##location of training data
studentNumber=10                                                    ##number of students to train from
studentRange=range(1,11)
faceRange=range(1,4)                                                        ##number of faces per student

"""get eigenvalues and corresponding eigenfaces"""
faceFrame=loadFaces(studentRange,faceRange)                        ##assemble the N number of 10304x1 vectors into a 10304xN matrix 
a, pseudoCov, averageVec=pseudoCovariance(faceFrame)
eigVals, eigFaces=eigen(a,pseudoCov)

"""find average unique baseline weight vector for the known faces"""
baseFrame, baseWeights=baselineCoords(faceFrame,eigVals,eigFaces,studentRange,faceRange)

"""train"""
trainStudent=range(1,11)
trainRange=range(4,8)
radiusFrame,maxDist=train(trainStudent,trainRange,dir_path,averageVec,eigFaces,baseWeights)

"""testing, graphic pop-up window, see below for statstics"""
rightGuess=0
totalCount=0
for i in range(0,numOfTestImages):
    totalCount+=1
    
    testStudent=int(np.random.randint(low=1,high=42,size=1))
    testPic=int(np.random.randint(low=8,high=10,size=1))

    testPath=dir_path.format(testStudent,testPic)
    testImg=cv2.imread(testPath,0)
    #cv2.imshow("test img",testImg)


    testVec,projectedFace,spaceDist,distVec=projectFace(testPath,averageVec,eigFaces,baseWeights)
    pos=np.argmin(distVec)+1

    distRelax=1.3

    if spaceDist>maxDist*distRelax:
        print("Actual student {}. Guessed ".format(testStudent) + "not a face")
        path=r"NOT_FACE-01.jpg"
        message=cv2.imread(path,0)
        initPos=414
        collage(dir_path,studentRange,testImg,projectedFace,message,initPos)
        guess=0
    else:
        minDist=distVec.min()
        possibleStudent="student {}".format(np.argmin(distVec)+1)
        studentRadius=radiusFrame[possibleStudent].values
        if minDist<studentRadius:
            print("Actual student {}. Guessed ".format(testStudent)+"face belongs to ", possibleStudent)
            path=r"pointer.jpg"
            message=cv2.imread(path,0)
            initPos=92*(pos-1)
            collage(dir_path,studentRange,testImg,projectedFace,message,initPos)
            guess=pos
        else:
            diffPercent=(minDist-studentRadius)/minDist
            if diffPercent<threshold:
                print("Actual student {}. Guessed ".format(testStudent)+"face belongs to ", possibleStudent)
                path=r"pointer.jpg"
                message=cv2.imread(path,0)
                initPos=92*(pos-1)
                collage(dir_path,studentRange,testImg,projectedFace,message,initPos)
                guess=pos
            else:
                print("Actual student {}. Guessed ".format(testStudent)+"new face")
                path=r"new-01.jpg"
                message=cv2.imread(path,0)
                initPos=414
                collage(dir_path,studentRange,testImg,projectedFace,message,initPos)
                guess=-1
    if guess==testStudent:
        rightGuess+=1
    if guess==-1 and 10<testStudent<41:
        rightGuess+=1
        
    cv2.waitKey(durationPerImg)
cv2.destroyAllWindows()
print("accuracy: ",round(rightGuess/totalCount*100,2),"%")


#       

        

