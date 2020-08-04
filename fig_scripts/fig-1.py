import cv2
import numpy as np
import math
noised = cv2.imread('1.png')
clean = cv2.imread('0.png')
noise = (noised/255)-(clean/255)
clean_gray = cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY)


# def gamma_trans(img, gamma):  # gamma函数处理
#     gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
#     gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
#     return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。
# mean = np.mean(clean_gray)
# gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
# print(gamma_val)
# texture = gamma_trans(img=clean_gray, gamma=gamma_val)

blur = cv2.GaussianBlur(clean,(21,21),0)

# sobelx = cv2.Sobel(clean_gray,cv2.CV_64F, 1, 0, ksize=3)
# sobely = cv2.Sobel(clean_gray, cv2.CV_64F, 0, 1, ksize=3)
sobelXY = cv2.Sobel(blur, cv2.CV_64F, 1, 1, ksize=3)

# cv2.imshow('nd', noised)
# cv2.imshow('n', noise)
# cv2.imshow('c', clean)
# cv2.imshow('s', blur)
# cv2.imshow('a', sobelXY)
# cv2.waitKey()

process=[noised, noise, clean, blur, sobelXY]
for i, t in enumerate(process):
    tmp = t[200:500,200:500,:]
    if i == 1:
        cv2.imshow('t',tmp)
        cv2.waitKey()
        cv2.imwrite(str(i)+'_.png', tmp)
