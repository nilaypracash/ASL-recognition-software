import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset= 20
imgSize = 300

folder = "Data/C"
counter = 0


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        #put one matrix into another (one picture into another)


        aspectRatio = h/w

        if aspectRatio > 1:
            constant = imgSize/h
            wCalc = math.ceil(constant*w)
            imgResize = cv2.resize(imgCrop,(wCalc,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCalc)/2)
            imgWhite[:, wGap:wCalc+wGap] = imgResize
        else:
            constant = imgSize / h
            hCalc = math.ceil(constant * h)
            imgResize = cv2.resize(imgCrop, (hCalc, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCalc) / 2)
            imgWhite[:, hGap:hCalc + hGap] = imgResize


        #output
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow('frame', img)
    #if s is pressed, take photo
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f"{folder}\Image_{time.time()}.jpg", imgWhite)
        counter = counter + 1
        print(counter)
