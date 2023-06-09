import cv2
import numpy as np

# 画像読み込み
img = cv2.imread('blended_image_90.jpg')

# RGB形式からBGR形式への変換
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 赤色のピクセルを検出
red_pixels = (img[:,:,0] > 200) & (img[:,:,1] < 50) & (img[:,:,2] < 50)

# 赤色のピクセルの座標を取得
y_indices, x_indices = np.where(red_pixels)

# 座標をnumpy配列として保存
np.savez('red_curve_coordinates', x=x_indices, y=y_indices)

coordinates = np.load('red_curve_coordinates.npz')
x_indices = coordinates['x']
y_indices = coordinates['y']

# 再生成用の画像を初期化 (黒背景)
reconstructed_img = np.zeros([3360, 6720, 3], dtype=np.uint8)

# 赤い曲線のピクセルを再生成
reconstructed_img[y_indices, x_indices] = [255, 0, 0]

cv2.imwrite('reconstructed_image.jpg', reconstructed_img)