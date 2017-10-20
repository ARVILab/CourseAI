import glob
from scipy.spatial import distance
from pylab import *
datasetRoot = '../../datasets/facs/'

k = 0

meanLandmark = np.load('meanCoords.npy')

for landmarkFile in glob.glob(datasetRoot+'landmarks/*/*/*.txt'):
    landmarks = np.loadtxt(landmarkFile)
    imgPath = landmarkFile.replace('_landmarks.txt', '.png').replace('landmarks', 'images')

    left = np.mean(landmarks[36:42], axis=0).astype(np.int32)
    right = np.mean(landmarks[42:48], axis=0).astype(np.int32)

    mPoint = (left + right)/2
    landmarks -= mPoint
    d = distance.euclidean(left, right)
    landmarks /= d

    landmarks -= meanLandmark
    np.save(landmarkFile.replace('.txt','_norm.npy'), landmarks)

    # for (x, y) in landmarks:
    #     plot(x, y, '+')
    # show()

    # img[left[1]-3:left[1]+3,left[0]-3:left[0]+3,2] = 255
    # img[right[1] - 3:right[1] + 3, right[0] - 3:right[0] + 3, 2] = 255
    # cv2.imwrite('test/'+str(k)+'.png', img)
    k += 1
    print k
