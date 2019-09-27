import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import os

img = cv2.imread("img.jpg")
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS )   

h_min=0;l_min=190;s_min=0
h_max=255;l_max=255;s_max=255    
h_min = np.array((h_min, l_min, s_min), np.uint8)
h_max = np.array((h_max, l_max, s_max), np.uint8)

thres = cv2.inRange(hls, h_min, h_max)
result_thres = cv2.bitwise_and(img,img, mask= thres)
black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
black_zone = cv2.rectangle(black,(0,470),(1024,740),(255, 255, 255), -1)
gray = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)               
ret,black_mask = cv2.threshold(gray,127,255, 0)

#prefinalmod = cv2.bitwise_and(result_thres,result_thres,mask = gray)

finalmod = cv2.bitwise_and(result_thres,result_thres,mask = black_mask)

#cv2.imshow('1',prefinalmod)
#cv2.imshow('fin', finalmod)

down_thres = 50
top_thres = 150
region = cv2.Canny(finalmod, down_thres, top_thres)
region = cv2.GaussianBlur(region, (3, 3),0)
hough_res = 1  
hough_ang = np.pi / 180
thres = 10 
min_line_length = 20  
max_line_gap = 7
line_image = np.copy(img) * 0
lines = cv2.HoughLinesP(region, hough_res, hough_ang, thres, np.array([]),min_line_length, max_line_gap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)

lines_region = cv2.addWeighted(img, 0.8, line_image, 1, 0)

cv2.imshow('result', lines_region)
cv2.imwrite('img_out.jpg', lines_region)
cv2.waitKey(0)

videoPath = os.getcwd()+'/video.mp4'
videoCapture = cv2.VideoCapture(videoPath)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(3)),
        int(videoCapture.get(4)))
videoWriter = cv2.VideoWriter('video_out.avi',
              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

down_thres = 50
top_thres = 150
hough_res = 1  
hough_ang = np.pi / 180  
thres = 15 
min_line_length = 50  
max_line_gap = 5

while True:
    succes, img = videoCapture.read()
    if succes == True:

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)    
        h_min=0;l_min=158;s_min=0
        h_max=255;l_max=255;s_max=255    
        h_min = np.array((h_min, l_min, s_min), np.uint8)
        h_max = np.array((h_max, l_max, s_max), np.uint8)
        thresh = cv2.inRange(hls, h_min, h_max)
        result_thres = cv2.bitwise_and(img,img, mask= thresh)

        black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        black_zone = cv2.rectangle(black,(0,500),(1024,739),(255, 255, 255), -1)
        gray = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)               
        ret,black_mask = cv2.threshold(gray,127,255, 0)
        finalmod = cv2.bitwise_and(result_thres,result_thres,mask = black_mask)
        
        region = cv2.Canny(finalmod, down_thres, top_thres)
        region = cv2.GaussianBlur(region, (3, 3),0)        
        line_image = np.copy(img) * 0
        lines = cv2.HoughLinesP(region, hough_res, hough_ang, thres, np.array([]),
                    min_line_length, max_line_gap)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)

        lines_region = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        cv2.imshow('result', lines_region)
        videoWriter.write(lines_region) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        exit()

videoCapture.release()
VideoWriter.release()
cv2.destroyAllWindows()