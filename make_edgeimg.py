import cv2
import numpy as np
def empty(x):
    pass
cv2.namedWindow('Parameters')
cv2.resizeWindow('Parameters', width = 600, height = 200)
cv2.createTrackbar('k_size_set', 'Parameters', 0, 10, empty)
cv2.createTrackbar('thresh_1', 'Parameters', 0, 500, empty)
cv2.createTrackbar('thresh_2', 'Parameters', 0, 500, empty)

# 画像を読み込む
img = cv2.imread('input_back.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Gaussian bulr
kernel = cv2.getTrackbarPos('k_size_set', 'Parameters')
kernel = (kernel * 2) + 1
img_blur = cv2.GaussianBlur(img_gray, (kernel, kernel), None)

med_val = np.median(img_blur)
sigma = 0.99  # 0.33
"""min_val = int(max(0, (1.0 - sigma) * med_val))
max_val = int(max(255, (1.0 + sigma) * med_val))"""

img_edge1 = cv2.Canny(img_blur, threshold1 = 30, threshold2 = 40)
# 画像を保存する
cv2.imwrite('edge_image_401_back.jpg', img_edge1)