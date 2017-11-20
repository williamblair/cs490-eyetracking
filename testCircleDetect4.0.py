# following along with tango with code guide
import numpy as np
import cv2
import sys

#Pupili tracking works!
#This code also averages all the circles together to get a more accurate read  on the pupil
#This code tracks pupil using circle tracking

# load the webcam
cam = cv2.VideoCapture(0)

# load the classifiers
faceCascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_alt.xml')
eyeCascade  = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml')

def getLeftEye(eyes):
    i = 0
    leftMost = 99999999
    leftMostIndex = -1
    #print(eyes)
    #print(eyes[0][0])
    #print(eyes[0][1])
    #print(eyes[1][0])

    while True:
        if i > len(eyes) - 1:
            break

        if (eyes[i][0] < leftMost):
            leftMost = eyes[i][0]
            leftMostIndex = i
        i = i + 1
    # print(leftMostIndex)

    if len(np.asarray(eyes).shape) != 1:
        #print("Eye chose", eyes[leftMostIndex])
        #print("Index", leftMostIndex)
        return eyes[leftMostIndex]


def getAverageCircle(circles):
    i = 0
    tempX = 0
    tempY = 0
    tempRadius = 0
    print("orig coords" , circles)
    while True:
        if i > len(circles) - 1:
            break

        tempX = tempX+circles[0][i][0]
        tempY = tempY+circles[0][i][1]
        tempRadius = tempRadius+circles[0][i][2]

        i = i + 1
    # print(leftMostIndex)
    print("before", tempX, tempY, tempRadius)
    tempX = tempX/(len(circles))
    tempY = tempY/(len(circles))
    tempRadius = tempRadius/(len(circles)+1)
    print("after", tempX, tempY, tempRadius)

    pupil = [tempX,tempY,tempRadius]

    print("Pupil before return", pupil)

    return  pupil

# function to figure out the location of eyes
def detectFaces(frame, faceCascade):
    # convert the frame and store it in grayscale (if necessary)
    if len(frame.shape) != 2:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = frame
    # equalize the image (enhance contrast) and store it back into itself
    cv2.equalizeHist(grayscale, grayscale)
    # get a list of faces (in rect form)
    faces = faceCascade.detectMultiScale(grayscale, 1.1, 2)

    return faces

def detectEyes(frame, eyeCascade):

    # convert the frame and store it in grayscale (if necessary)
    if len(frame.shape) != 2:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = frame
    # equalize the image (enhance contrast) and store it back into itself
    # cv2.equalizeHist(grayscale, grayscale)
    # get a list of eyes (in rect form)
    #eyes = eyeCascade.detectMultiScale(grayscale, 1.1, 2)

    eyes = eyeCascade.detectMultiScale(grayscale)
    eyes = getLeftEye(eyes)
    return eyes

def detectIrises(frame):
    # convert to grayscale if necessary
    if len(frame.shape) != 2:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = frame
    # increase contrast
    cv2.equalizeHist(grayscale, grayscale)
    # look for circles
    #print 'eye shape: ', grayscale.shape

    # in tango's code, he uses
    # minDist = eye.cols / 8 --> grayscale.shape[1] / 8
    # threshold = 250
    # minArea = 15
    # minRadius = eye.rows / 8 --> grayscale.shape[0] / 8
    # maxRadius = eye.rows / 3 --> grayscale.shape[0] / 3

    # from stack overflow:
    # param1 - the first method specific parameter. In the
    # case of CV_HOUGH_GRADIENT, it is the higher threshold
    # of the two passed to the Canny() edge detector (the
    # lower one is twice smaller)
    # param2 - second method specific parameter. In the case
    # of CV_HOUGH_GRADIENT, it is the accumulator threshold
    # for the circle centers at the detection stage. the
    # smaller it is, the more false circles may be detected.
    # circles, corresponding to the larger accumulator values,
    # will be returned first

    minDist = grayscale.shape[1] / 8
    minRadius = grayscale.shape[0] / 8
    maxRadius = grayscale.shape[0] / 3

    param1 = 250
    param2 = 15

    irises = cv2.HoughCircles(grayscale, cv2.HOUGH_GRADIENT,
                              1,minDist,param1=param1,param2=param2,
                              minRadius=minRadius,maxRadius=maxRadius)

    if not (irises is None):
        irises = getAverageCircle(irises)

    return irises

# what if instead of using houghcircles, we just
# draw a circle in the center of the eye square?
# edit - because then you can't move the eyeball
# left, right, up, down, etc.
#def getCenters(eyes):


# empty image placeholder
clip = np.zeros((512,512,3), np.uint8)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)

    if not ret:
        print 'Error reading frame!'
        sys.exit(-1)

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get the rectangles defining the face
    faces = detectFaces(gray, faceCascade)

    # iterate through each detected face
    for face in faces:
        # draw a rectangle around each face
        cv2.rectangle(frame, (face[0],face[1]), (face[0]+face[2],face[1]+face[3]), (255,0,0), 2)

        # crop the face to only look there for eyes
        fx = face[0]
        fy = face[1]
        fw = face[2]
        fh = face[3]
        #face_cropped = gray[fy:fy + fh, fx:fx + fw]
        face_cropped = gray[fy:fy+fh, fx:fx+fw]

        # detect eyes from the cropped face
        eyes = detectEyes(face_cropped, eyeCascade)

        # draw squares around the eyes
        if not (eyes is None):
            #for eye in eyes:
                ex = eyes[0]
                ey = eyes[1]
                ew = eyes[2]
                eh = eyes[3]
                #cv2.rectangle(face_cropped, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)
                # draw a rectangle around the found eye
                cv2.rectangle(frame, (fx+ex,fy+ey), (fx+ex+ew,fy+ey+eh),(0,255,0),2)

                # calculate the middle half of the detected eye square (to
                # focus on just the eye more)
                # c stands for cropped
                cex = ex
                cey = ey + (eh / 4)
                cew = ew
                ceh = 2 * eh / 4

                # clip out just the eye section
                eye_cropped = face_cropped[ey:ey+eh, ex:ex+ew]
                #eye_cropped = face_cropped[cey:cey+ceh, cex:cex+cew]

                # detect circles within the eye crop
                irises = detectIrises(eye_cropped)

                # go through each iris
                if not (irises is None):
                        irises = np.uint16(np.around(irises))
                    #for iris in irises[0,:]:
                        print(irises)
                        ix = irises[0]
                        iy = irises[1]
                        ir = irises[2]

                        #cv2.circle(frame, (int(fx+cex+ix),int(fy+cey+iy)), ir, (0,0,255), 2)
                        #cv2.circle(frame, (fx + cex + ix, fy + cey + iy), ir, (0, 0, 255), 2)
                        cv2.circle(frame, (fx + ex + ix, fy + ey + iy), ir, (0, 0, 255), 2)

                # testing eye clipping
                clip = eye_cropped

        # checking cropped image
        #clip = face_cropped



    '''eyes = detectEyes(frame, eyeCascade)
    # iterate through each eye if they exist
    if not (eyes is None):
        for eye in eyes:
            # get the eye rect x,y,w,h
            ex = eye[0]
            ey = eye[1]
            ew = eye[2]
            eh = eye[3]

            # draw a rect around the eye
            cv2.rectangle(frame, (ex, ey), (ew, eh), (0, 0, 255), 2)
            # cv2.rectangle(clip, (ex, ey), (ew, eh), (0, 0, 255), 2)


    for circle in circles[0,:]:
        # draw the outer circle
        cv2.circle(frame, (circle[0],circle[1]),circle[2],(0,255,0),2)

    # go through each detected eye
    for x,y,w,h in eyes:
        # draw a rectangle around the eye
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        # cut out the eye section
        roi_color = frame[y:y+h, x:x+w]
        roi_gray  = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

        # detect circles within the eye
        circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT,
                                   1,20,param1=50,param2=30,
                                   minRadius=0,maxRadius=0)
        # get the x y and radius of each detected circle
        if not (circles is None):
            for circle in circles:
                ex = int(circle[0][0])
                ey = int(circle[0][1])
                er = int(circle[0][2])
                cv2.circle(frame, (x+ex,y+ey), er, (255,0,0), 2)
                #print 'circle: ', circle
    '''



    # show the frame
    cv2.imshow('frame', frame)
    # show the blurred version of the frame
    frame = cv2.medianBlur(frame, 5)
    cv2.imshow('frame blurred', frame)
    # test what cv2.histogram looks like
    if len(clip.shape) == 2:
        cv2.equalizeHist(clip, clip)

    cv2.imshow('clipped', clip)

    # test cutting out the eye only

    # quit on keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close the cam
cam.release()
cv2.destroyAllWindows()
