from keras.models import load_model
from time import sleep
from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'C:/Users/arish/Desktop/dataset/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Dataset Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier(r'C:\Users\arish\Desktop\Expression_detection\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\arish\Desktop\Expression_detection\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

cap = cv2.VideoCapture(0)
file1 = open("e.txt","r+") 
a=file1.read()
print(a)

while True:
    _, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))

    

        for (x,y,w,h) in faces:
          #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
          roi_gray = gray[y:y+h,x:x+w]
          roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
         


          if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            print(label)
            
            if ((confidence > 82) and ( label== a )):
              cv2.putText(frame,'authorized',(40,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
              cv2.putText(frame,'not authorized',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
         # else:
         #    cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
       
        pass

    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()