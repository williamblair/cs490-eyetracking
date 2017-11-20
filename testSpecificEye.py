import numpy as np
import cv2

#This code was the first attempt at tracking one eye (not pupil) Forget this code, testCircle1.0 is better version of same thing
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml')
x = 1
y = 1
w = 1
h = 1

tempCam = cap.read()

def getLeftEye(eyes):
    i=0
    leftMost = 99999999
    leftMostIndex = -1

    while True:
        if i >= len(eyes)-1:
            break

        if (eyes[0][0] < leftMost):
            leftMost = eyes[0][0]
            leftMostIndex = i

        i = i+1
    #print(leftMostIndex)
    return eyes[leftMostIndex]

def isEmpty(eyes):
    if eyes == ():
        return False
    else:
        getLeftEye(eyes)
    return True

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]

    roi_color = frame[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
 

    #print(eyes)
    if isEmpty(eyes):
        #for (ex, ey, ew, eh) in eyes:
        eyeRect = getLeftEye(eyes)
        cv2.rectangle(roi_color, (eyeRect[0], eyeRect[1]), (eyeRect[0]+eyeRect[2],eyeRect[1]+eyeRect[3]), (0, 255, 0), 2)
        #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)



    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()