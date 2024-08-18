import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
sub_data = 'Shreya'  # Inside main folder
path = os.path.join(datasets, sub_data)  # datasets/Trump-gets added in the directory

if not os.path.isdir(path):  # If Trump does not exist in that dir then create a new dir
    os.makedirs(path)  # Creates the path Trump if it does not exist

(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)  # Loading algorithm

webcam = cv2.VideoCapture(0)  # Cam id (primary/secondary) cam initializer

count = 1
while count <= 50:  # 50 pics captured
    print(count)
    (_, im) = webcam.read()  # Reading frame from cam
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]  # Crop only the bounding box part
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
    count += 1

    cv2.imshow("OpenCV", im)
    key = cv2.waitKey(10)
    if key == 27:  # Press 'ESC' to break the loop
        break

webcam.release()
cv2.destroyAllWindows()
