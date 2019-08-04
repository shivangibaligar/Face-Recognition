import cv2
cam = cv2.VideoCapture(0)
detector= cv2.CascadeClassifier('C:/Users/Shivangi/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

Id=input('enter your id')     
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #returns a rectange with face 
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)  #scale factor decreases the size of bigger images during training by 32% and then reshapes it later on
                                                                               #minNeighbors specifies the min. neighbors to be detected as true positive 
                                                                                    #lesser no. more false positive
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
        
        sampleNum=sampleNum+1
        
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame',img)
        #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
