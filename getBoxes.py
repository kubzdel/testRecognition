import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

def standarizeName(x,y,dim):
    xText = str(x)
    yText = str(y)
    maks = len(str(dim))
    while(x%40!=0):
        x+=1
    while (y % 40 != 0):
        y+=1
    rightPad = maks - len(xText)
    padding1 = ""
    for i in range(0, rightPad):
        padding1 += '0'
    rightPad2 = maks - len(yText)
    padding2 = ""
    for i in range(0, rightPad2):
        padding2 += '0'

    return padding1+str(x)+"_"+padding2+str(y)



BORDER_SIZE = 300
for file in os.listdir("scans"):
    img_rgb = cv2.imread("scans/"+file)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    x_dim, y_dim = img_gray.shape
    template = cv2.imread('indicator.png',0)
    template =  cv2.resize(template, (58, 43))

    w, h = template.shape[:2]
    w = 130
    h = 350
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    outputImage = cv2.copyMakeBorder(
        img_gray,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
                     cv2.BORDER_CONSTANT,
                     value=(255,255,255)
                  )

    mask = np.zeros(outputImage.shape[:2], np.uint8)

    newDir = os.getcwd() + "/samples/" + file.split('.')[0]
    if not os.path.exists(newDir):
        os.makedirs(newDir)

    for pt in zip(*loc[::-1]):
        if mask[pt[1] + int(h / 2), pt[0] + int(w / 2)] != 255:
            mask[pt[1]:pt[1] + h, pt[0]:pt[0] + w] = 255
            lst = list(pt)
            lst[0] =lst[0]-50+BORDER_SIZE
            lst[1] +=(55+BORDER_SIZE)
            pt = tuple(lst)

            cv2.rectangle(outputImage, pt, (pt[0] + 130, pt[1] + 350), (0,0,255), 2)
            crop_img = outputImage[pt[1]:pt[1]+350,pt[0]:pt[0]+130]
           # cv2.imshow("cropped", crop_img)
            name = standarizeName(pt[0],pt[1],max(x_dim,y_dim))
            crop_img = cv2.resize(crop_img,None,fx=0.4,fy=0.4)
            ret, crop_img = cv2.threshold(crop_img, 170, 255, cv2.THRESH_BINARY)
            cv2.imwrite(newDir+"/sample_"+name+".png",crop_img)

cv2.imwrite('res.png',outputImage)