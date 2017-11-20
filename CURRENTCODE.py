# following along with tango with code guide
import numpy as np
import cv2
import sys
import win32api
import msvcrt
#This code uses calibration to decide a box size to scale against
# constants
# SCREEN_WIDTH  = 1920
# SCREEN_HEIGHT = 1080
calibrationMode = True
clibrationCorner = 1
readInput = False
topLeft = [0.0,0.0]
bottomRight = [0.0,0.0]
centerCalc = [0.0,0.0]
center = []

SCREEN_WIDTH = win32api.GetSystemMetrics(0)
SCREEN_HEIGHT = win32api.GetSystemMetrics(1)

MOUSE_SCALE_X = 20
MOUSE_SCALE_Y = -30

# load the webcam
cam = cv2.VideoCapture(0)

# load the classifiers
faceCascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_alt.xml')
eyeCascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml')

# holds the previous center
lastPoint = [0, 0]

# holds the location of the mouse
mousePoint = [SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2]


def getLeftEye(eyes):
    i = 0
    leftMost = 99999999
    leftMostIndex = -1

    while True:
        if i > len(eyes) - 1:
            break
        if (eyes[i][0] < leftMost):
            leftMost = eyes[i][0]
            leftMostIndex = i

        i = i + 1
    # print(leftMostIndex)
    if not (eyes is None):
        if len(np.asarray(eyes).shape) != 1:
            return eyes[leftMostIndex]
        else:
            return eyes
    else:
        return None


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
    # eyes = eyeCascade.detectMultiScale(grayscale, 1.1, 2)
    eyes = eyeCascade.detectMultiScale(grayscale)

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
    # print 'eye shape: ', grayscale.shape

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
                              1, minDist, param1=param1, param2=param2,
                              minRadius=minRadius, maxRadius=maxRadius)

    return irises


# what if instead of using houghcircles, we just
# draw a circle in the center of the eye square?
# edit - because then you can't move the eyeball
# left, right, up, down, etc.
# def getCenters(eyes):

# looks for the blackest circle
def getEyeBall(frame, circles):
    # stores the total pixel sum values for each circle
    sums = np.zeros(len(circles))

    # print frame
    # print circles

    # loop through the frame's rows
    for y in range(frame.shape[0]):
        # loop through that row (length of the row):
        for x in range(frame.shape[1]):
            # loop through each circle
            for i in range(len(circles)):
                # get the center point of the circle
                center = (int(circles[0][i][0]), int(circles[0][i][1]))
                radius = int(circles[0][i][2])

                # checks if the pixel is inside the circle, and
                # if so adds it to the total circle values
                if (pow(x - center[0], 2) + pow(y - center[1], 2) < pow(radius, 2)):
                    sums[i] += frame[y][x]

    # figure out the smallest sum
    smallestSum = 9999999
    smallestIndex = -1
    for i in range(len(circles)):
        if sums[i] < smallestSum:
            smallestIndex = i

    return circles[0][smallestIndex]


# get the last average X amount of circle locations
# points is the list of circle points, amount
# is the number of points to average
def stabilize(points, amount):
    sumX = 0
    sumY = 0
    count = 0
    for i in xrange(max(0, len(points) - amount), len(points)):
        sumX += points[i][0]  # x
        sumY += points[i][1]  # y
        count += 1
    if count > 0:
        sumX /= count
        sumY /= count

    return (sumX, sumY)


# empty image placeholder
clip = np.zeros((512, 512, 3), np.uint8)

# holds the past centers of the eyeball, to use for averaging
# centers = []

# hold previous mouse points
mousePoints = []

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # keyPressed  = msvcrt.getch()
    # if keyPressed == 'x':
    #    print 'x was pressed'

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
        cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)

        # crop the face to only look there for eyes
        fx = face[0]
        fy = face[1]
        fw = face[2]
        fh = face[3]
        # face_cropped = gray[fy:fy + fh, fx:fx + fw]
        face_cropped = gray[fy:fy + fh, fx:fx + fw]

        # detect eyes from the cropped face
        eyes = detectEyes(face_cropped, eyeCascade)

        # draw squares around the eyes
        if not (eyes is None):

            # test getting left eye
            eyes = np.asarray(getLeftEye(eyes))
            # print eyes

            if (not (eyes is None)) and not len(eyes) == 0:
                eye = eyes
                # for eye in eyes:
                # for ex, ey, ew, eh in eyes:
                ex = eye[0]
                ey = eye[1]
                ew = eye[2]
                eh = eye[3]
                # cv2.rectangle(face_cropped, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)

                # draw a rectangle around the found eye
                cv2.rectangle(frame, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), (0, 255, 0), 2)

                # print('Eye frame: X: ' + str(ex) + ',' + str(ex+ew))
                # print('Eye frame: Y: ' + str(ey) + ',' + str(ey+eh))

                # calculate the middle half of the detected eye square (to
                # focus on just the eye more)
                # c stands for cropped
                cex = ex
                cey = ey + (eh / 4)
                cew = ew
                ceh = 2 * eh / 4

                # clip out just the eye section
                eye_cropped = face_cropped[ey:ey + eh, ex:ex + ew]
                # eye_cropped = face_cropped[cey:cey+ceh, cex:cex+cew]

                # detect circles within the eye crop
                irises = detectIrises(eye_cropped)

                # go through each iris
                if not (irises is None):
                    irises = np.uint16(np.around(irises))
                    '''for iris in irises[0,:]:
                        ix = iris[0]
                        iy = iris[1]
                        ir = iris[2]

                        #cv2.circle(frame, (int(fx+cex+ix),int(fy+cey+iy)), ir, (0,0,255), 2)
                        #cv2.circle(frame, (fx + cex + ix, fy + cey + iy), ir, (0, 0, 255), 2)
                        cv2.circle(frame, (fx + ex + ix, fy + ey + iy), ir, (0, 0, 255), 2)'''

                    # get the one that is the darkest and call
                    # it the eyeball
                    eyeball = getEyeBall(eye_cropped, irises)
                    # print 'eye cropped shape: ', eye_cropped.shape
                    # print eyeball
                    # add the current center to the list of centers
                    # centers.append((eyeball[0], eyeball[1]))
                    # calculate the new average center
                    # center = stabilize(centers, 5) # use the past 5 entries
                    center = (eyeball[0], eyeball[1])

                    # calculate the new mouse position
                    if (cv2.waitKey(1) & 0xFF == ord(' ')):
                        readInput = True

                    if (clibrationCorner == 1):
                        print "Look at the top left corner and press the space bar"
                    if (readInput == True and clibrationCorner == 1):
                        topLeft = center
                        clibrationCorner = 2
                        readInput = False
                        print "Look at the bottom right corner and press the space bar"

                    if (readInput == True and clibrationCorner == 2):
                        bottomRight = center
                        clibrationCorner = 3
                        readInput = False

                    if clibrationCorner == 3:
                        print topLeft, bottomRight
                        #print SCREEN_HEIGHT,SCREEN_WIDTH
                        calibrationMode = False
                        centerCalc[0] = (long(topLeft[0]) - long(bottomRight[0])) / 2
                        centerCalc[1] = (long(topLeft[1]) - long(bottomRight[1])) / 2
                        ew = long(bottomRight[0]) - long(topLeft[0])
                        eh = long(topLeft[1]) - long(bottomRight[1])
                        MOUSE_SCALE_X = SCREEN_WIDTH/ew
                        MOUSE_SCALE_Y = SCREEN_HEIGHT/eh
                        clibrationCorner = 11
                        print centerCalc, ew, eh

                    if calibrationMode == False:
                        if not (center is None):
                            diff = [0, 0]
                            # diff[0] = (center[0] - lastPoint[0]) * MOUSE_SCALE_X
                            # diff[1] = (center[1] - lastPoint[1]) * MOUSE_SCALE_Y
                            #print(MOUSE_SCALE_X)
                            #print(MOUSE_SCALE_Y)
                            diff[0] = (long(center[0])-long(centerCalc[0])) * MOUSE_SCALE_X
                            diff[1] = (long(center[1])-long(centerCalc[1])) * MOUSE_SCALE_Y
                            #print long(centerCalc[0]) - long(center[0])
                            ##print long(centerCalc[1]) - long(center[1])
                            #print (long(centerCalc[0]) - long(center[0])) * MOUSE_SCALE_X
                            #print (long(centerCalc[1]) - long(center[1])) * MOUSE_SCALE_Y
                            print (long(center[0]) - long(centerCalc[0]))
                            print (long(center[1]) - long(centerCalc[1]))
                            print center[0], center[1], centerCalc[0], centerCalc[1], diff[0], diff[1]

                            # print 'Diff: ', diff
                            mousePoint[0] = diff[0]
                            mousePoint[1] = diff[1]
                            # move the mouse the difference
                            '''mousePoint[0] += diff[0]
                            mousePoint[1] += diff[1]
                            if( mousePoint[0] > SCREEN_WIDTH ):
                                mousePoint[0] = SCREEN_WIDTH
                            elif( mousePoint[0] < 0):
                                mousePoint[0] = 0
                            if (mousePoint[1] > SCREEN_HEIGHT):
                                mousePoint[1] = SCREEN_HEIGHT
                            elif (mousePoint[1] < 0):
                                mousePoint[1] = 0
                            '''

                            # append to the list of mouse points
                            #mousePoints.append((SCREEN_WIDTH * center[0] / ew, SCREEN_HEIGHT * center[1] / eh))

                            # average the last 5 mouse points
                            #mousePoint = stabilize(mousePoints, 5)
                            # print mousePoint, mousePoints
                            # ratio: center_x / eye_width = mouse_x / screen_width
                            # mousePoint[0] = SCREEN_WIDTH * center[0] / ew
                            # ratio: center_y / eye_height = mouse_y / screen_height
                            # mousePoint[1] = SCREEN_HEIGHT * center[1] / eh
                            # print 'Mousepoint: ', mousePoint
                            win32api.SetCursorPos((mousePoint[0], mousePoint[1]))

                            lastPoint = center
                            cv2.rectangle(frame, (fx + ex + topLeft[0], fy + ey + topLeft[1]), (fx + ex + ew, fy + ey + eh), (0, 255, 0), 2)




                            # draw the eyeball circle
                            # if not (eyeball is None):
                            # cv2.circle(frame, (fx + ex + eyeball[0], fy + ey + eyeball[1]), eyeball[2], (0,0,255), 2)
                            # cv2.circle(frame, (fx+ex+center[0],fy+ey+center[1]), eyeball[2], (0,0,255), 2)
                    # cv2.circle(frame, (fx + ex + center[0], fy + ey + center[1]), 10, (0, 0, 255), 2)
                    cv2.circle(frame, (fx + ex + center[0], fy + ey + center[1]), 2, (0, 0, 255), 2)


                    # set the current center as the last point
                    # lastPoint = center

                    # testing eye clipping
                    clip = eye_cropped

                    # checking cropped image
                    # clip = face_cropped

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
