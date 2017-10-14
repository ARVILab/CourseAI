import os
import dlib
from imutils import face_utils
from scipy.misc import imread, imsave
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

faces_dataset = '../datasets/facs/'

img_paths = [os.path.join(root, f)
             for root, _, files in os.walk(faces_dataset + 'images')
             for f in files if f.endswith('.png')]

txt_paths = [os.path.join(root, f)
             for root, _, files in os.walk(faces_dataset + 'emotions')
             for f in files if f.endswith('.txt')]

emotions = set()
for p in txt_paths:
    s = float(open(p).read().strip())
    emotions.add(s)

print(emotions)

for p in img_paths:
    im = imread(p, mode='RGB')

    # detect faces in the grayscale frame
    rects = detector(im, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(im, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        print(shape.shape)
        for (x, y) in shape:
            cv2.circle(im, (x, y), 1, (0, 0, 255), -1)

        imsave('test.png', im)