import cv2
import dlib
import numpy as np

def to_np_arr(landmarks, dtype="int"):

	coords = np.zeros((68, 2), dtype=dtype)

	for i in range(0, 68):

		coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

	return coords

def rotate_bound(image, angle):
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



def rect_to_bb(face_rect):
    x = face_rect.left()
    y = face_rect.top()
    w  = face_rect.right() - x
    h = face_rect.bottom() - y
    return (x,y,w,h)



def get_right_eye(landmarks, dtype = "int"):

    cords = np.zeros((6,2),dtype = dtype)
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
    angle = np.degrees(np.arctan(dy,dx)) - 180

    return angle

cap = cv2.VideoCapture()
# Creating face detection and landmark predictor objects using dlib
face_detect = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
glasses = cv2.imread('sunglasses.png')


while True:
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame',frame)
    else:
        print("Nope")
        break



# while True:
#     ret, frame = cap.read()
#     if ret == True:
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         faces = face_detect(gray)
#         for face in faces:
#             landmarks = landmark_predict(face,gray)
#             right_eye_pts = get_right_eye(landmarks)
#             left_eye_pts = get_left_eye(landmarks)
#             inclination = get_inclination_angle(right_eye_pts,left_eye_pts)
#
#
#         glasses = rotate_bound(glasses,inclination)
#         cv2.imshow('Frame', frame)
#         cv2.imshow('Inclined glasses',glasses)
#     else:
#         print("Program build unsucessful\n")
#         break
#     key = cv2.waitKey(1) & 0xff
#     if key == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
