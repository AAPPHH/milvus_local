import cv2
import numpy as np

def calculate_histogram_similarity(img1, img2):
    hist_size = 256
    ranges = [0, 256]
    scores = []

    for i in range(3):  # B, G, R
        hist1 = cv2.calcHist([img1], [i], None, [hist_size], ranges)
        hist2 = cv2.calcHist([img2], [i], None, [hist_size], ranges)

        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        scores.append(score)

    avg_bhatt = np.mean(scores)
    similarity_percent = 100 * (1 - avg_bhatt)
    similarity_percent = max(0, min(similarity_percent, 100))

    return round(similarity_percent, 2)

def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    hist_similarity = calculate_histogram_similarity(img1, img2)
    print(f"Farbliche Ähnlichkeit (Histogramm - Bhattacharyya): {hist_similarity}%")

# Beispielnutzung
path1 = r"C:\Users\jfham\OneDrive\Desktop\milvus_local\images_v2\image_data\coco2017_train\train2017\000000000034.jpg"
path2 = r"C:\Users\jfham\OneDrive\Desktop\milvus_local\images_v2\image_data\coco2017_train\train2017\000000000395.jpg"
compare_images(path1, path2)
