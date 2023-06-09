import cv2
import numpy as np

# 画像の読み込みと初期化
background = np.zeros((3360, 6720, 3), dtype=np.uint8)
result = None

for i in range(0, 9):
    # 画像を読み込む
    #img = cv2.imread('images/edge_sample/edge_sample_rotate_%d.jpg' % i)
    foreground = cv2.imread('images/edge_sample/edge_sample_rotate_%d.jpg' % i, cv2.IMREAD_UNCHANGED)
    alpha = foreground[:, :, 3] / 255.0


    # 初めての画像の場合はそのまま代入し、それ以降は重ね合わせる
    if result is None:
        result = background.astype(np.float64)
    else:
        # 画像を重み付けして合成する
        alpha = 1.0 / (i + 1)  # 重みの計算
        # 赤い曲線を黒背景に重ね合わせる
        result = cv2.addWeighted(result, 1 - alpha, foreground[:, :, :3].astype(np.float64), alpha, 0)

# 結果を正規化して整数値に変換する
result = result.astype(np.uint8)

# 合成結果を保存する
cv2.imwrite('blended_image.jpg', result)
