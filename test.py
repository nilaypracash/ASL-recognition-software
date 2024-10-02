import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
offset= 20
imgSize = 300

folder = "Data/C"
counter = 0

labels = ["A","B","C"]


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        #out one matrix into another (one picture into another)


        aspectRatio = h/w

        if aspectRatio > 1:
            constant = imgSize/h
            wCalc = math.ceil(constant*w)
            imgResize = cv2.resize(imgCrop,(wCalc,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCalc)/2)
            imgWhite[:, wGap:wCalc+wGap] = imgResize
            prediction, index =classifier.getPrediction(imgWhite)
            print(prediction,index)
        else:
            constant = imgSize / h
            hCalc = math.ceil(constant * h)
            imgResize = cv2.resize(imgCrop, (hCalc, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCalc) / 2)
            imgWhite[:, hGap:hCalc + hGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)





        prediction, index =classifier.getPrediction(imgWhite)
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255))

        #output
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow('frame', imgOutput)

    cv2.waitKey(1)
