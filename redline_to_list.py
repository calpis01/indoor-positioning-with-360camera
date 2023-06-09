import cv2
import numpy as np
import pickle
# 1. 画像の読み込み
img = cv2.imread('blended_image_90.jpg')

# 2. 色に基づいて赤い曲線を検出する
red_channel = img[:,:,2] # 画像から赤色チャンネルを取得
_, red_lines = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY) # 閾値を設定して赤い曲線を抽出

# 3. 検出された曲線を元にエッジ検出を行う
edges = cv2.Canny(red_lines, 100, 200) # Cannyエッジ検出器を使用

# 4. エッジ検出された座標を正規化し、直線のパラメータを計算する
y_indices, x_indices = np.where(edges > 0) # エッジのピクセル座標を取得

# 画像の中心を原点とした座標系に変換
x_indices = x_indices - img.shape[1] / 2
y_indices = img.shape[0] / 2 - y_indices

# 球面座標系に変換
theta = 2 * np.pi * x_indices / img.shape[1] # 経度
phi = np.pi * (y_indices / img.shape[0] - 0.5) # 緯度

with open('file.txt', 'w') as f:
    print(theta, file=f)
# ここで、thetaとphiは各エッジのピクセルに対応する直線のパラメータになります
print(np.get_printoptions())
print(theta.shape, phi.shape)