import numpy as np
import cv2

#Original eye tracking cascade file
#Shitty eye selection but atleast the camera works

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml')
x = 1
y = 1
w = 1
h = 1

tempCam = cap.read()


def getLeftEye(eyes):
    i = 0
    leftMost = 99999999
    leftMostIndex = -1

    while True:
        if i >= len(eyes) - 1:
            break
        if (eyes[i][0] < leftMost):
            leftMost = eyes[i][0]
            leftMostIndex = i

        i = i + 1
    # print(leftMostIndex)
    return eyes[leftMostIndex]


def isEmpty(eyes):
    if eyes == ():
        return False
    else:
        getLeftEye(eyes)
    return True

def getLeftCircle(Circles):
    i = 0
    leftMost = 99999999
    leftMostIndex = -1

    while True:
        if i >= len(circles) - 1:
            break
        if (circles[0][0][0] < leftMost):
            leftMost = eyes[0][0][0]
            leftMostIndex = i

        i = i + 1
    # print(leftMostIndex)
    return circles[leftMostIndex]

def hasCircles(circles):
    if circles is None:
        return False
    elif circles == ():
        return False

    else:
        getLeftCircle(circles)
        return True

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]
    cv2.equalizeHist(roi_gray,roi_gray)

    roi_color = frame[y:y + h, x:x + w]
#Start Eye finding
    eyes = eye_cascade.detectMultiScale(roi_gray)
    eyeRect = eyes
    circles = ()
    # print(eyes)
    if isEmpty(eyes):
        eyeRect = getLeftEye(eyes)
        #circles = np.uint16(np.around(circles))
        # for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (eyeRect[0], eyeRect[1]), (eyeRect[0] + eyeRect[2], eyeRect[1] + eyeRect[3]),
                      (0,255,0), 2)
        # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

#Start circle finding
        focusEye = frame[eyeRect[1]:eyeRect[1] + eyeRect[3], eyeRect[0]:eyeRect[0] + eyeRect[2]]
        focusEye = cv2.cvtColor(focusEye, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(focusEye,focusEye)
        circles = cv2.HoughCircles(focusEye, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0,
                                   maxRadius=30)
        print(eyeRect[1]) #Y
        print(eyeRect[0]) #X
        showEye = frame[eyeRect[1]:eyeRect[1]+50, eyeRect[0]:eyeRect[0]+50]

    if hasCircles(circles):
        print(circles)
        cv2.circle(showEye, (int(circles[0][0][0])+eyeRect[0], int(circles[0][0][1])+eyeRect[1]), circles[0][0][2], (0, 255, 0), 2)

#circles[0][0][0] = x, [0][0][1] = y, [0][0][2] = radius

    cv2.imshow('frame', frame)
    cv2.imshow('eyes', showEye)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

circles = ()
eyes = ()
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()