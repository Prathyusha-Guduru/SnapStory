#Importing necessary libraries

import cv2
import numpy as np
import dlib
from collections import OrderedDict
import imutils




#To get co-ordinates for rectangles around the eyes
def eye_roi(face_roi):
    ptl = []
    face_gray_roi = cv2.cvtColor(face_roi,cv2.COLOR_BGR2GRAY)
    eyes = eye_classifier.detectMultiScale(face_gray_roi)
    for (ex,ey,ew,eh) in eyes:
        ptl.append((ex,ey,ew,eh))
    return ptl

#Functions to get the inclination angle

face_landmark = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

def get_right_eye(landmarks, dtype = "int"):

    cords = np.zeros((6,2) , dtype = dtype)
    for i in range(36,42):
        cords[i-36] = (landmarks.part(i).x , landmarks.part(i).y)
    return cords

def get_left_eye(landmarks, dtype = "int"):

    cords = np.zeros((6,2),dtype = dtype)
    for i in range(42,48):
        cords[i-42] = (landmarks.part(i).x , landmarks.part(i).y)
    return cords


def get_inclination_angle(right_eye_pts,left_eye_pts):
    right_eye_center  = right_eye_pts.mean(axis = 0).astype("int")
    left_eye_center = left_eye_pts.mean(axis=0).astype("int")

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan(dy, dx)) - 180

    return (angle)

def to_np_arr(landmarks, dtype="int"):

	coords = np.zeros((68, 2), dtype=dtype)

	for i in range(0, 68):

		coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

	return coords


def rotate_bound(image,angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))





# Creating face detection and landmark predictor objects using dlib
face_detect = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



#Haar caascades for face detection and eye detection
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier  = cv2.CascadeClassifier('haarcascade_eye.xml')

#Passing the input imge
img = cv2.imread('Hans-solo.jpg')
glasses = cv2.imread('CroppedGlasses.jpg')

#Converting the color image to gray scale for easy and fast computation
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Detecting faces and getting the respective ROI's
faces = face_classifier.detectMultiScale(img_gray,1.3,5)


faces_dlib = face_detect(img_gray)

for face in faces_dlib:
    landmarks = landmark_predict(img_gray, face)
    landmarks = to_np_arr(landmarks)
    (lStart, lEnd) = face_landmark["left_eye"]
    (rStart, rEnd) = face_landmark["right_eye"]
    leftEyePts = landmarks[lStart:lEnd]
    rightEyePts = landmarks[rStart:rEnd]

    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    angle_glass = rotate_bound(glasses, angle)
    # angle_glass = cv2.rotate(glasses,angle)
    # angle_glass = imutils.rotate(glasses, angle)

for (x,y,w,h) in faces:
    #Drawing rectangle around face
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #Face ROI
    roi_color = img[y:y+h, x:x+w]
    #Getting eye co-ordinates
    ptl =eye_roi(roi_color)
    pt1 = (ptl[0][0]-10,ptl[0][1]-5)
    pt2 = (ptl[1][0]+ptl[1][2]+10,ptl[1][1]+ptl[1][3]+5)
    #Drawing a rectangle around both the eyes
    #cv2.rectangle(roi_color,pt1,pt2,(0,255,0),2)
    #Getting the eye roi
    eye_roi_img  = roi_color[ptl[0][1]-5:ptl[1][1]+ptl[1][3]+5,ptl[0][0]-10:ptl[1][0]+ptl[1][2]+10]

#Creating backgrund and foreground for masking /overlay
bg = eye_roi_img
fg = angle_glass
kernel = np.ones((3,3),np.uint8)

w,h = bg.shape[:2]
fg = cv2.resize(src = fg, dsize=(h,w),interpolation=cv2.INTER_NEAREST)
# fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
fg = cv2.dilate(fg,kernel,iterations = 2)
# fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)


mask = cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)
mask = cv2.bitwise_not(mask)


final = cv2.bitwise_and(fg,bg)

roi_color[ptl[0][1]-5:ptl[1][1]+ptl[1][3]+5,ptl[0][0]-10:ptl[1][0]+ptl[1][2]+10] = final



cv2.imshow('img', img)
cv2.imshow('face', roi_color)
cv2.imshow('angleglasses', angle_glass)
cv2.imshow('eyeroi', eye_roi_img)
cv2.imshow('mask', mask)
cv2.imshow('BG',bg)
cv2.imshow('FG',fg)
cv2.imshow('Finale',final)
cv2.waitKey(0)
cv2.destroyAllWindows()



