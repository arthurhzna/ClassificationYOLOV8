from ultralytics import YOLO

import numpy as np

model = YOLO("C:\\Users\\62853\\Downloads\\Mediapipe\\best1.pt")  # load a custom model

results = model("C:\\Users\\62853\\Downloads\\Mediapipe\\20240211_210458_jpg.rf.16c890efa569008120582bf7f56aa7b6.jpg")  # predict on an image


names_dict = results[0].names

probs = results[0].probs.data.tolist()

max_value = probs[0]
max_index = 0

for i in range(1, len(probs)):
    # Bandingkan nilai elemen saat ini dengan nilai tertinggi yang tersimpan
    if probs[i] > max_value:
        # Jika nilai elemen saat ini lebih besar, perbarui nilai tertinggi dan indeks
        max_value = probs[i]
        max_index = i

# print("a")
# print(names_dict)
# print("b")
# print(probs)
# print("KLASIFIKASI")
print("KLASIFIKASI", names_dict[np.argmax(probs)])
# print("PROBABILITAS")
print("PROBABILITAS:", max_value)