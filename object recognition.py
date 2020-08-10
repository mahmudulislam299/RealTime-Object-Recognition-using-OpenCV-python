#Mahmudul Islam
#EEE,BUET'15
#Email: mahmudulislam299@gmail.com

import cv2
import numpy as np
MIN_MATCH_COUNT=30
detector=cv2.xfeatures2d.SIFT_create()
#detector=cv2.SIFT()



FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

flannParam2=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann2=cv2.FlannBasedMatcher(flannParam2,{})

flannParam3=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann3=cv2.FlannBasedMatcher(flannParam3,{})

#flannParam2=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
#flann2=cv2.FlannBasedMatcher(flannParam2,{})

trainImg=cv2.imread("book1.jpg",0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
trainImg2=cv2.imread("book2.jpg",0)
trainKP2,trainDesc2=detector.detectAndCompute(trainImg2,None)
trainImg3=cv2.imread("book3.jpg",0)
trainKP3,trainDesc3=detector.detectAndCompute(trainImg3,None)
#trainImg3=cv2.imread("book3.jpg",0)
#trainKP3,trainDesc3=detector.detectAndCompute(trainImg3,None)

cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)
    matches2=flann2.knnMatch(queryDesc,trainDesc2,k=2)
    matches3=flann3.knnMatch(queryDesc,trainDesc3,k=2)
    #matches2=flann2.knnMatch(queryDesc,trainDesc2,k=2)

    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        cv2.putText(QueryImgBGR, "Electronic Circuit",(150,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5,(0,0,255),3)
        
    else:
        print "Not Enough match found book1- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
    cv2.imshow('result',QueryImgBGR)
    

    goodMatch2=[]
    for m2,n2 in matches2:
        if(m2.distance<0.75*n2.distance):
            goodMatch2.append(m2)
    if(len(goodMatch2)>MIN_MATCH_COUNT):
        tp2=[]
        qp2=[]
        for m2 in goodMatch2:
            tp2.append(trainKP2[m2.trainIdx].pt)
            qp2.append(queryKP[m2.queryIdx].pt)
        tp2,qp2=np.float32((tp2,qp2))
        H2,status=cv2.findHomography(tp2,qp2,cv2.RANSAC,3.0)
        h,w=trainImg2.shape
        trainBorder2=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder2=cv2.perspectiveTransform(trainBorder2,H2)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder2)],True,(0,255,0),5)
        cv2.putText(QueryImgBGR, "Electric Machine",(150,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5,(0,0,255),3)
    else:
        print "Not Enough match found book2- %d/%d"%(len(goodMatch2),MIN_MATCH_COUNT)
    cv2.imshow('result',QueryImgBGR)


    goodMatch3=[]
    for m3,n3 in matches3:
        if(m3.distance<0.75*n3.distance):
            goodMatch3.append(m3)
    if(len(goodMatch3)>MIN_MATCH_COUNT):
        tp3=[]
        qp3=[]
        for m3 in goodMatch3:
            tp3.append(trainKP3[m3.trainIdx].pt)
            qp3.append(queryKP[m3.queryIdx].pt)
        tp3,qp3=np.float32((tp3,qp3))
        H3,status=cv2.findHomography(tp3,qp3,cv2.RANSAC,3.0)
        h,w=trainImg3.shape
        trainBorder3=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder3=cv2.perspectiveTransform(trainBorder3,H3)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder3)],True,(0,255,0),5)
        cv2.putText(QueryImgBGR, "Electromagnetics",(150,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5,(0,0,255),3)
    else:
        print "Not Enough match found book3- %d/%d"%(len(goodMatch3),MIN_MATCH_COUNT)
        
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
 
