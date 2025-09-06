# datacollect.py
# pip install opencv-python==4.12.0 

import cv2
import os

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

id = input("Enter Your ID: ")
name = input("Enter Your Name: ")

if not os.path.exists("users.txt"):
    with open("users.txt", "w") as f:
        f.write(f"{id},{name}\n")
else:
    exists = False
    with open("users.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(id + ","):
            exists = True
            break
    if not exists:
        with open("users.txt", "a") as f:
            f.write(f"{id},{name}\n")

count = 0

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite('datasets/User.' + str(id) + "." + str(count) + ".jpg",
                    gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if count > 500:
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done")