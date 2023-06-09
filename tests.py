import cv2
import numpy as np

redline_image = cv2.imread('images/output/output_rotate_0.jpg')
if redline_image is None:
    print(f'Could not open or find the image: images/output/output_rotate_0.jpg')
# 90枚の画像を合成する
for i in range(1, 90):
    logo_image = cv2.imread('images/output/output_rotate_%i.jpg' % i)
    rows, cols, channels = logo_image.shape

    roi = redline_image[0:rows, 0:cols]
    # ロゴの画素のRGB値をマスクを用いて0にする
    logo_mask = cv2.cvtColor(logo_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(logo_mask, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    logo_fg = cv2.bitwise_and(logo_image, logo_image, mask=mask)

    if i == 1:
        cv2.imwrite('logo_fg.jpg', logo_fg)
        cv2.imwrite('logo_bg.jpg', roi_bg)
    else:
        dst = cv2.add(roi_bg, logo_fg)
        redline_image[0:rows, 0:cols] = dst

cv2.imwrite('blended_image_90_half.jpg', redline_image)
