import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
sub_data = 'Shreya'  # Subfolder inside the main folder
path = os.path.join(datasets, sub_data)  # datasets/Trump gets added in the directory

# Create the subfolder if it doesn't exist
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)  # Load the Haar cascade file

webcam = cv2.VideoCapture(0)  # Initialize the webcam (0 is usually the default camera)

count = 1
while count < 51:  # Capture 50 images
    print(count)
    ret, im = webcam.read()  # Read a frame from the webcam
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # Detect faces in the image

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a rectangle around the face
        face = gray[y:y + h, x:x + w]  # Crop the face from the image
        face_resize = cv2.resize(face, (width, height))  # Resize the face image
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)  # Save the face image
        count += 1

        # Display the name of the person
        cv2.putText(im, sub_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("OpenCV", im)  # Show the image with the detected face
    key = cv2.waitKey(10)  # Wait for a key press for 10 ms
    if key == 27:  # Exit if the 'Esc' key is pressed
        break

webcam.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
