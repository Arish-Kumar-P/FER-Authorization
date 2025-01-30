from keras.models import load_model
from time import sleep
from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

face_classifier = cv2.CascadeClassifier(r'C:\Users\arish\Desktop\Expression_detection\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\arish\Desktop\Expression_detection\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

cap = cv2.VideoCapture(0)
count = 0
c=0

while True:
    _, frame = cap.read()


    if face_extractor(frame) is not None:
        if c==1:
         count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        if count<=50 :
         file_name_path = 'C:/Users/arish/Desktop/dataset/'+str(count)+'.jpg'

        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        #cv2.imshow('Face Cropper',face)
    else:
        print("Face not found")
        pass
     
    
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            label_position1= (x,y-50)
            if c==0:
             cv2.putText(frame,'"c" for capture expression',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(16,78,139),2)
            if ((cv2.waitKey(1) & 0xFF == ord('c')) and (c==0)):
                file1 = open("e.txt","w")
                file1.write(label)
                c=1
            if count<=48 and c==1:    
             cv2.putText(frame,'wait for accuracy',label_position1,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)   
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,64,64),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        if count>=50:
            cv2.putText(frame,'q for quit',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count==70 :
        break

cap.release()
cv2.destroyAllWindows()