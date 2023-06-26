import numpy as np
import cv2
def calculate_intersect(image1, image2):
    intersection = np.logical_and(image1, image2)
    intersection_score = np.sum(intersection)
    return intersection_score

def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    iou_score = np.sum(intersection)/ 1435578
    return iou_score
def create_overlay_image(edge_image, curve_image):
    # 曲線画像をエッジ画像と同じ形状にリサイズ
    resized_curve_image = cv2.resize(curve_image, (edge_image.shape[1], edge_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 二つの画像を加算して重ね合わせる
    overlay_image = cv2.addWeighted(edge_image, 0.5, resized_curve_image, 0.5, 0)

    return overlay_image

intersection_rank = []
edge_image = cv2.imread('images/new_edge/lab.jpg', 0)
for i in range(0,45):
    curve_image = cv2.imread(f'images/output/output_rotate_{i}.jpg', 0)
    resized_curve_image = cv2.resize(curve_image, (edge_image.shape[1], edge_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    intersection_score = calculate_iou(edge_image, resized_curve_image)
    intersection_rank.append((intersection_score, i))  

intersection_rank.sort(key=lambda x: x[0], reverse=True)
print(np.sum(np.logical_or(edge_image, resized_curve_image)))
cv2.imwrite('images/curve.jpg', resized_curve_image)
# Display the top 5 images
for rank, (score, idx) in enumerate(intersection_rank[:5], 1):
    print(f"Rank {rank}, Intersection score: {score}, Original index: {idx}")
    curve_image = cv2.imread(f'images/output/output_rotate_{idx}.jpg',0)  # idxに対応する曲線画像を取得するコードを適宜書く
    overlay_image = create_overlay_image(edge_image, curve_image)
    cv2.imwrite(f'images/overlay_rank{rank}.jpg', overlay_image*255)

