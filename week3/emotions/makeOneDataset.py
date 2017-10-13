import glob
import os
import numpy as np
import cv2
datasetRoot = '/DATA/CourseAI/datasets/facs/'
X = []
Y = []
k = 0
for personId in os.listdir(datasetRoot + 'images/'):
    for clipId in os.listdir(datasetRoot + 'images/' + personId):
        x_img = []
        x_land = []
        clipPath = datasetRoot + 'images/' + personId+'/' + clipId + '/'

        facs_label_path = glob.glob(clipPath.replace('images', 'labels') + '*.npy')
        if len(facs_label_path):
            facs_label_path = facs_label_path[0]
            facs_label = np.load(facs_label_path)

            for imgPath in sorted(glob.glob(clipPath + '*.png')):
                landPath = imgPath.replace('images', 'landmarks').replace('.png', '_landmarks_norm.npy')
                landmark = np.load(landPath)
                img = cv2.resize(cv2.imread(imgPath), (224, 224), interpolation=cv2.INTER_CUBIC)
                x_img.append(img)
                x_land.append(landmark)
            x_img = np.array(x_img)
            x_land = np.array(x_land)

            X.append([x_img, x_land])


            em_label_path = glob.glob(clipPath.replace('images', 'emotions') + '*.npy')
            if len(em_label_path) > 0:
                em_label = np.load(em_label_path[0])
            else:
                em_label = None

            Y.append(np.array([facs_label, em_label]))
            k+=1
            print k


np.save('/DATA/CourseAI/datasets/facs/X.npy', X)
np.save('/DATA/CourseAI/datasets/facs/Y.npy', Y)
print('done')