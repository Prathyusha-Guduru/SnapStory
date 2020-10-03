import  cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt


glasses = cv2.imread('classicglasses.jpg')
glasses_gray = cv2.cvtColor(glasses,cv2.COLOR_BGR2GRAY)

glasses_inv = cv2.bitwise_not(glasses_gray)
bg = cv2.imread('random_bg.jpg')
# glasses_inv = cv2.resize(src = glasses_inv, bg.shape,interpolation=cv2.INTER_NEAREST)
#glasses_inv = glasses_inv.reshape(glasses.shape)
print(glasses_inv.shape)

w,h = glasses.shape[:2]
#
# glasses_inv = cv2.resize(src = glasses_inv, dsize=(h,w),interpolation=cv2.INTER_NEAREST)
print(glasses_inv.shape)
fg = cv2.bitwise_or(glasses,glasses,mask = glasses_inv)

# bg = np.ones(glasses.shape,dtype = np.uint8)
# bg[:,:] = [0,0,255]

bg = cv2.imread('Hans-solo.jpg')
bg = cv2.resize(src = bg, dsize=(h,w),interpolation=cv2.INTER_NEAREST)


final = cv2.bitwise_or(bg,fg)


while True:
    cv2.imshow('Original img',glasses)
    cv2.imshow('Inverse', glasses_inv)
    cv2.imshow('grayscale', glasses_gray)
    cv2.imshow('bg',bg)
    cv2.imshow('fg',fg)
    cv2.imshow('Final',final)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break


