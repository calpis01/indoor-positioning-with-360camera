import cv2
import numpy as np

# 90枚の画像に対応する座標を保存するリスト
all_coordinates = {}

for i in range(10):
    # 画像読み込み
    img = cv2.imread(f'/images/output/output_rotate_{i}.jpg')

    # RGB形式からBGR形式への変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 赤色のピクセルを検出
    red_pixels = (img[:,:,0] > 200) & (img[:,:,1] < 50) & (img[:,:,2] < 50)

    # 赤色のピクセルの座標を取得
    y_indices, x_indices = np.where(red_pixels)

    # 座標をnumpy配列として保存
    np.savez(f'saved_lines/red_curve_coordinates_{i}', x=x_indices, y=y_indices)

    coordinates = np.load(f'saved_lines/red_curve_coordinates_{i}.npz')
    x_indices = coordinates['x']
    y_indices = coordinates['y']

    # 再生成用の画像を初期化 (黒背景)
    reconstructed_img = np.zeros([3360, 6720, 3], dtype=np.uint8)

    # 赤い曲線のピクセルを再生成
    reconstructed_img[y_indices, x_indices] = [255, 0, 0]
    
    cv2.imwrite(f'images/reconstructed/reconstructed_image_{i}.jpg', reconstructed_img)

      # 座標を辞書として保存
    all_coordinates[str(i)] = {'x': x_indices, 'y': y_indices}

# 辞書をnumpy配列として保存
np.savez('all_coordinates', **all_coordinates)
