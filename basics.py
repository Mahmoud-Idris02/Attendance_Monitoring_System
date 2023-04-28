"""
@project    Attendance_Monitoring
@file_name  basics.py
@Author     Mahmoud Alaa Eldeen Fathy
@email      malaafathy02@gmail.com
@linkedin   https://www.linkedin.com/in/mahmoud-alaa-149859242/
@GitHub     https://github.com/Mahmoud-UL
"""


#********************** Import section s **********************************
import cv2
import numpy as np
import face_recognition
#********************** Import section e **********************************


#step1 loadig images and converting it to RGB from BGR
imgElon = face_recognition.load_image_file("Pictures/Elon Musk.jpg")
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("Pictures/Elon Musk2.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#finding the faces in the image then finding their encodings as well
faceLoc = face_recognition.face_locations(imgTest)[0]  #this function returns 4 values top ,byttom,right and left
print(faceLoc)
encodeTest =face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)

faceLoc = face_recognition.face_locations(imgElon)[0]
print(faceLoc)
encodeElon =face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)

# comparing faces and find the distance between them
results = face_recognition.compare_faces([encodeElon],encodeTest) # we can give here a list of known faces instead of [encodedElon]
faceDis = face_recognition.face_distance([encodeElon],encodeTest) # calculating  distance to find the best match
print("comparison? ",results)
print("face distance= ",faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Musk2',imgTest)
cv2.waitKey(0)



