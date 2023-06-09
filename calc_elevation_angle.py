import numpy as np
import cv2
def calculate_intersect(image1, image2):
    intersection = np.logical_and(image1, image2)
    intersection_score = np.sum(intersection)
    return intersection_score

def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    iou_score = np.sum(intersection)/ np.sum(union)
    return iou_score

intersection_rank = []
edge_image = cv2.imread('edge_image.jpg', 0)
for i in range(0,90):
    curve_image = cv2.imread(f'images/output/output_rotate_{i}.jpg', 0)
    intersection_score = calculate_iou(edge_image, curve_image)
    intersection_rank.append((intersection_score, i))  

intersection_rank.sort(key=lambda x: x[0], reverse=True)

# Display the top 5 images
for rank, (score, idx) in enumerate(intersection_rank[:5], 1):
    print(f"Rank {rank}, IoU score: {score}, Original index: {idx}")

