import numpy as np
import cv2

inconNames = np.load('../../datasets/icons/X.npy')
X = []
for icon_name in inconNames:
    img = cv2.resize(cv2.imread('../../datasets/icons/smallIcons/' + icon_name + '.png'), (64,64), interpolation=cv2.INTER_CUBIC)
    X.append(img[:, :, :3])

x_train = np.asarray(X)
np.save('../../datasets/icons/allicons64.npy', x_train)
print('done')