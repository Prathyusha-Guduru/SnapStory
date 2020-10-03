#Importing all the necessary libraries: 
import cv2
import numpy as np
import dlib
import math


def get_nose(landmarks, dtype = "int"):

    cords = np.zeros((8,2),dtype = dtype)
    for i in range(29, 36):
        cords[i-29] = (landmarks.part(i).x, landmarks.part(i).y)
    return cords


cap = cv2.VideoCapture(0)
face_detect = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
pig_nose = cv2.imread('Pig_nose.png')


while True:
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detect(gray)
        for face in faces:
            landmarks = landmark_predict(gray, face)
            nose = get_nose(landmarks)

            top_nose = (landmarks.part(29).x, landmarks.part(29).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_nose = (landmarks.part(32).x, landmarks.part(32).y)
            right_nose = (landmarks.part(35).x, landmarks.part(35).y)

            #nose_width = int(math.hypot(left_nose[0]-right_nose[0], left_nose[1]-right_nose[1]))
            nose_width = int(math.hypot(left_nose[0] - right_nose[0],
                                   left_nose[1] - right_nose[1]) *1.7)

            nose_height = int(nose_width * 0.7)

            top_left = (int(center_nose[0]-nose_width/2),int(center_nose[1]-nose_height/2))
            bottom_right = (int(center_nose[0]+nose_width/2),int(center_nose[1]+nose_height/2))

            pig_nose = cv2.resize(pig_nose,(nose_height,nose_width))
            pig_nose_gray = cv2.cvtColor(pig_nose,cv2.COLOR_BGR2GRAY)
            _,  thresh = cv2.threshold(pig_nose_gray,25,255,cv2.THRESH_BINARY_INV)

            nose_area = frame[top_left[1] : top_left[1] + nose_height,top_left[0] : top_left[0] + nose_width]
            nose_area_mask = cv2.bitwise_and(nose_area, nose_area, mask = thresh)

            final_nose = cv2.add(nose_area_mask, pig_nose)
            frame[top_left[1]: top_left[1] + nose_height,
            top_left[0]: top_left[0] + nose_width] = final_nose



        cv2.imshow('Frame',frame)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
