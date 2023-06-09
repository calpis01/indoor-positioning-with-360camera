import numpy as np
import cv2

# npzファイルの読み込み
loaded_coordinates = np.load('red_curve_coordinates.npz', allow_pickle=True)

# 再生成したい画像のキー
key = 9

# x, y座標を取得
coordinates = loaded_coordinates[str(key)].item()
x_coords = coordinates['x']
y_coords = coordinates['y']

# 画像サイズを指定
w, h = 6720, 3360

# 指定された大きさの空の画像（黒色）を生成
image = np.zeros((h, w, 3), dtype=np.uint8)

# 指定の座標に赤色を配置
image[y_coords, x_coords] = [0, 0, 255]  # 注意: OpenCVでは色の順序がBGRなので赤色は[0, 0, 255]

# 画像を保存
cv2.imwrite('reconstructed_image_'+str(key)+'.jpg', image)