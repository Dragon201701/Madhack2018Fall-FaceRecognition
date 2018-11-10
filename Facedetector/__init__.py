import numpy as np
import cv2
import sys
import sysconfig
from matplotlib import pyplot as plt
from numpy import imag
from cv2 import waitKey

def welcome():
    print("Hello, World!")
def camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
def facedetect():
    #imagepath = sys.argv[1]
   
    image = cv2.imread(r'E:/faces2.jpg')
    cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("input_image", image)
    waitKey(0)
    face_cascade = cv2.CascadeClassifier('C:\opencv3.4.1\sources\data\haarcascades\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.15,minNeighbors = 5,minSize = (5,5),flags = cv2.CASCADE_SCALE_IMAGE)
    #print ("detect {0} faces",format(len(faces))
    
    divisor=8
    h, w =np.shape(image)[:2]
    minSize=(w/divisor, h/divisor)
    color = (0,0,0)
    numfaces = 0
    if len(faces)>0:
        for faceRect in faces:  
            print('[INFO] Detecting face number ' + str(numfaces))
            x, y, w, h = faceRect
            cv2.rectangle(image, (x, y), (x+w, y+h), color)
            roi = image[y:(h+y), x:(w+x)]
            cv2.imshow('facerect' + str(numfaces), roi)
            cv2.imwrite('face' + str(numfaces) + '.jpg', roi)
            numfaces = numfaces + 1
            
    cv2.imshow('img',image)
    #cv2.imshow('facerect',roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    welcome()
    #facedetect()
    camera()
    
    