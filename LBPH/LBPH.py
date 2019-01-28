
import numpy as np
import pandas as pd
import cv2

def load(path):
    """loads an image, find its height and width, creates an empty image that is 
    2 pixels less than the original image in both directions. Binary feature
    information will be written to this new empty image. Returns the loaded image,
    trimmed image, x-direction extent, and y-direction extent"""
    
    img=cv2.imread(path,0)
    height,width=img.shape
    xEnd=height-1
    yEnd=width-1
    newImg=np.zeros((xEnd-1,yEnd-1))
    return img, newImg, xEnd, yEnd

def binary (i,j,img,newImg):
    """performs the binary operation defined by Ojala. see details online"""
    
    """find the 8 pixel neighborhood around the ceneter pixel defined by coordinates (x=i, y=j)"""
    center=img[i,j]
    up=(i-1,j)
    down=(i+1,j)
    left=(i,j-1)
    right=(i,j+1)
    upLeft=(i-1,j-1)
    upRight=(i-1,j+1)
    lowLeft=(i+1,j-1)
    lowRight=(i+1,j+1)

    neighborhood=(up,upRight,right,lowRight,down,lowLeft,left,upLeft)
    tempList=[]
    for k in neighborhood:                              #extracts the pixel info of the 8 neighbor pixels into an array
        tempList.append(img[k])
    neighborValues=np.array(tempList)
    
    
    binary=(neighborValues>center).astype(int)          #create a binary array based on the neighbor pixel values array from above
                                                        #tests if the surrounding pixels are larger or smaller than the center pixel. if larger, 1. smaller, 0
    decimal=np.packbits(binary)                         #the binary array is converted to a decimaldecimal value
    newImg[i-1,j-1]=decimal                             #assign the decimal value to the center pixel

def baseLine(path,studentIds,photoIds,gridX,gridY,writePath,faceNum):
    """find and collects the the characteristic histograms of the known subject faces into a singel dataframe
    note that each subject's characteristic histogram is an average taken over x faces, as defined by the number in variable photoIds
    writes the result to a text file, which will be read for fast subsequent training and use"""
    hold=pd.DataFrame()                                 #overall histogram dataframe
    for i in studentIds:                                #chose a subject
        temp=np.zeros((gridX*gridY*256,1)).flatten('F') #creates empty array for histogram addition and later for averaging
        for j in photoIds:                              #for each photo of the same subject
            directory=path.format(i,j)                  #creates file path
            img,newImg,xEnd,yEnd=load(directory)        #see load function
            for m in range(1,xEnd):                     #performs the binary operation on each of the pixels in the x and y direction
                for n in range(1,yEnd):
                    binary(m,n,img,newImg)

            newImg=newImg.astype(np.uint8)              #making sure that the datatype stays same
            vec=histo(gridX,gridY,newImg,xEnd,yEnd)     #performs histogram operation on the new, binary image. see histo function for more details
            temp+=vec                                   #summing the characteristic histogram
        average=temp/faceNum                            #averaging the characteristic histogram
        title="student{}".format(i)                     
        hold[title]=average                             #write to dataframe
    hold.to_csv(writePath,index=False)                  #writes the histograms to file 

def histo(gridX,gridY,newImg,xEnd,yEnd):
    """divides the newImg into a gridX by gridY grid. Performs histogram 
    operation on each of the subsequent grid cells. Concatenates all the
    cell histograms into a global histogram. Returns the global histogram
    """
    
    xIndex=np.arange(0,xEnd)                            
    yIndex=np.arange(0,yEnd)

    xBins=np.array_split(xIndex,gridX)                  #splits the x coordinates into the grids
    yBins=np.array_split(yIndex,gridY)                  #splits the y coordinates into grids

    globalHisto=[]                                      #concatenated histogram for entire img
    for m in range(0,len(xBins)):   
        for n in range(0,len(yBins)):
            segX=xBins[m]
            segY=yBins[n]

            subImg=newImg[segX[0]:segX[-1]+1,segY[0]:segY[-1]+1].flatten('F')
            hist=list(np.histogram(subImg,bins=np.arange(0,257))[0])
            globalHisto.extend(hist)
    globalHisto=np.array(globalHisto)
    return globalHisto

def readBase(path):
    """read the histogram of the known subjects produced from the baseLine function"""
    base=pd.read_csv(path)
    return base

def test(base,testId,testFace,loadPath,gridX,gridY):
    """finds the charactersitic histogram of a testing image and compares it to
    the collection of n known subjects' histograms from the baseLine function. Returns
    a vector with n distance values"""
    
    testPath=loadPath.format(testId,testFace)
    testImg,newTestImg,xEnd,yEnd=load(testPath)

    for i in range(1,xEnd):
        for j in range(1,yEnd):
            binary(i,j,testImg,newTestImg)

    testHisto=histo(gridX,gridY,newTestImg,xEnd,yEnd)
    testHisto=(np.matrix(testHisto)).T
    distVec=base-testHisto
    dist=np.linalg.norm(distVec,axis=0)
    return dist

def closeness(vec):
    """a metric that determines whether a test image belongs to the known subject
    group or whether the test image is a completely new face.
    It is observed that for test image which belong to the know subject group, the
    difference between smallest value and the second smallest value of the dist vector
    produced from the 'test' function above would be much larger than the same
    difference for a test image that does not belong to the known subject group.
    this function returns that difference
    """
    indexSort=np.argsort(vec)
    choice=indexSort[0]
    nearest=indexSort[1]
    closeness=abs((vec[choice]-vec[nearest])/vec[choice])*100
    return closeness

def train(base,testId,testFace,loadPath,gridX,gridY):
    """finds the 'closeness' metric for a collection of test images
    the metric is then used to find whats the smallest difference that determines
    whether an image belongs to a known subject or is a new face
    """
    metrix=[]
    for i in testId:
        for j in testFace:
            dist=test(base,i,j,loadPath,gridX,gridY)
            diff=closeness(dist)
            metrix.append(diff)
    return np.array(metrix)

def collage(testIds,loadPath,testImg,binaryImg,middlePath, ID):
    """for stitching together the pop-up window that shows the known subjects,
    the testing image,the extracted binary feature image, and the decisions regarding
    whether it is a known face (and then identify) or a new face. 
    
    Note necessary for operation of the identification. Purely for view
    """
    path=loadPath.format(testIds[0],1)
    firstBand=cv2.imread(path,0)
    for i in range(2,len(testIds)+1):
        path=loadPath.format(i,1)
        img=cv2.imread(path,0)
        firstBand=np.concatenate((firstBand,img),axis=1)
#
    height,width=firstBand.shape
    heightBinary,widthBinary=binaryImg.shape
    hStrip=np.zeros((1,widthBinary))
    vStrip=np.zeros((heightBinary+2,1))
    
    binaryImg=np.concatenate((hStrip,binaryImg,hStrip),axis=0)
    binaryImg=np.concatenate((vStrip,binaryImg,vStrip),axis=1)
    stitch=np.concatenate((testImg,binaryImg),axis=1)
    heightStitch,widthStitch=stitch.shape
    initBottom=int((width-widthStitch)/2) 
    bottomBand=np.zeros(firstBand.shape)    
    for i in range(0,widthStitch):
        bottomBand[:,initBottom]=stitch[:,i]
        initBottom+=1
#    firstBand=np.concatenate((firstBand,middleBand)).astype(np.uint8)
    
    middleBand=np.zeros(firstBand.shape)
    middleImg=cv2.imread(middlePath,0)
    heightMiddle,widthMiddle=middleImg.shape
    if ID>0:
        initMiddle=int(widthMiddle*(ID-1))
    else:
        initMiddle=int((width-widthMiddle)/2)
    for i in range(0,widthMiddle):
        middleBand[:,initMiddle]=middleImg[:,i]
        initMiddle+=1
        
    
    return np.concatenate((firstBand,middleBand,bottomBand)).astype(np.uint8)
