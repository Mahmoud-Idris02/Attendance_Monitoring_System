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
import os
from datetime import datetime

#********************** Import section e **********************************




#********************** Subprogram section s **********************************
def loadingImgs():
    # Loading images from path & getting their names
    path = "Pictures"
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    # print(classNames)
    return images,classNames

def findEncodings(images):
    encodelist = []
    for img in images:
        # converting to RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

from datetime import datetime

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        names = []
        for line in myDataList:
            entry = line.split(',')
            names.append(entry[0])
        if name not in names:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{dtString}\n')
            print(f'{name} marked as present at {dtString}')
        else:
            print(f'{name} has already been marked as present')





#********************** Subprogram section e **********************************

images,classNames = loadingImgs()
encodelistKnown = findEncodings(images)
# print("\"the length of the encoded list must equl to the #.images\" \n encodelist length= ",len(encodelistKnown))
print("Encoding completed")


# capturing the web cam img
cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    # reducing the size of captured image
    imgS =cv2.resize(img,(0,0),None,0.25,0.25)
    #converting to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    # Finding Matches
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodelistKnown,encodeFace)
        # it will give us a list of distances
        faceDis = face_recognition.face_distance(encodelistKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        #write the name of the detected pioneer
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            #drawing a rectangle around his face & displaying his name
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)






