import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
classifier = cv2.CascadeClassifier('C:\\Users\\Shivangi\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=classifier.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),thickness=2)
        faceId, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        #print(confidence)
        if(confidence<70):
            if(faceId==1):
                cv2.putText(im,"Shivangi", (x,y+h), font, 1, 255,2)
            elif(faceId==2):
                cv2.putText(im,"Vishesh", (x,y+h), font, 1, 255,2)
                
        else:
            cv2.putText(im,"Unknown", (x,y+h), font, 1, 255,2)
    cv2.imshow('im',im) 
    if cv2.waitKey(1) == 27:
        break
cam.release()
cv2.destroyAllWindows()
