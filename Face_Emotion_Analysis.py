#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image


# In[2]:


def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    face_haar_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')#Load haar classifier
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles
    return faces,gray_img


# In[ ]:


model = tf.keras.models.load_model('nana.h5')
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

cap=cv2.VideoCapture(0)
while True:
    ret,test_img=cap.read()          # captures frame and returns boolean value and captured image
    test_img = cv2.flip(test_img,1)
    faces_detected,gray_img=faceDetection(test_img)
    
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w, x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
    
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    resized_img = cv2.resize(test_img, (900, 600))
    cv2.imshow('Facial emotion analysis ',resized_img)

    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break
cap.release()
cv2.destroyAllWindows


# In[ ]:




