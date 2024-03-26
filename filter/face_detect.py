# import the necessary packages
from imutils import face_utils
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

image = cv2.imread('face_acne/11.jpeg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 0)

# print(rects)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 4)

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:

        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# show the output image with the face detections + facial landmarks
# cv2.imshow("Output", image)
cv2.imshow('Whitened Face', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # Detect faces in the image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
# # Brighten the detected faces
# for (x, y, w, h) in faces:
#     # Extract the region of interest (ROI) which is the face
#     face_roi = image[y:y + h, x:x + w]
#
#     # Increase brightness by scaling the face ROI
#     brightness_scale = 1.2
#     face_roi = cv2.convertScaleAbs(face_roi, alpha=brightness_scale, beta=0)
#
#     # Place the brightened face back into the original image
#     image[y:y + h, x:x + w] = face_roi
#
# # Display the result
# cv2.imshow('Brightened Faces', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# # Load the pre-trained face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # Load the image
# # image = cv2.imread('face.jpg')
#
# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Detect faces in the image
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
# # Brighten only the detected face regions
# for (x, y, w, h) in faces:
#     # Extract the region of interest (ROI) which is the face
#     face_roi = image[y:y + h, x:x + w]
#
#     # Convert the ROI to the LAB color space
#     lab_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
#
#     # Increase brightness by scaling the L channel
#     brightness_scale = 1.2
#     lab_roi[:, :, 0] = cv2.convertScaleAbs(lab_roi[:, :, 0], alpha=brightness_scale, beta=0)
#
#     # Convert the LAB ROI back to the BGR color space
#     face_roi_brightened = cv2.cvtColor(lab_roi, cv2.COLOR_LAB2BGR)
#
#     # Place the brightened face back into the original image
#     image[y:y + h, x:x + w] = face_roi_brightened
#
# # Display the result
# cv2.imshow('Brightened Faces', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
