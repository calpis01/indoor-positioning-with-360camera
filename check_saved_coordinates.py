import cv2
import numpy as np
# 保存された辞書を読み込む
loaded_coordinates = np.load('all_coordinates.npz', allow_pickle=True)

# 辞書内の各エントリを調べる
for key in loaded_coordinates.keys():
    print(f'Key: {key}')
    coordinates = loaded_coordinates[key].item()
    print(f'X coordinates: {coordinates["x"]}')
    print(f'Y coordinates: {coordinates["y"]}\n')

