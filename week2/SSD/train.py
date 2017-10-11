from ssd import SSD300
from ssd_utils import BBoxUtility
from ssd_training import MultiboxLoss
from pycocotools.coco import COCO
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import pickle

model = SSD300(num_classes=81)

mbLoss = MultiboxLoss(num_classes=81)
model.compile(loss=mbLoss.compute_loss, optimizer='adam')


ssd300_priors = pickle.load(open('prior_boxes_ssd300.pkl'))
bboxer = BBoxUtility(num_classes=81, priors=ssd300_priors)

cocodata = COCO('/DATA/COCO/annotations/instances_train2017.json')
cocoDir = '/DATA/COCO/train2017/'
catsToIds = {}

for i, catid in enumerate(cocodata.getCatIds()):
    catsToIds[catid] = i


def generator(batch_size=4):
    imgList = cocodata.imgs.keys()
    imgcount =len(imgList)
    n=0
    while True:
        X = np.zeros((batch_size, 300, 300, 3), dtype=np.float)
        Y = []
        for i in range(0, batch_size):
            n = n%imgcount
            img_data = cocodata.imgs[imgList[n]]
            img_path = cocoDir + img_data['file_name']
            img = imread(img_path, mode="RGB").astype('float32')
            X[i] = imresize(img, (300, 300)).astype('float32') / 127.5 - 1.
            anns = cocodata.imgToAnns[imgList[n]]
            bboxList = []
            for ann in anns:
                bbox = np.array(ann['bbox'], dtype=np.float)
                '''
                annotation{
                    "id" : int, 
                    "image_id" : int, 
                    "category_id" : int, 
                    "segmentation" : RLE or [polygon], 
                    "area" : float, 
                    "bbox" : [x,y,width,height], 
                    "iscrowd" : 0 or 1,
                }
                assign_boxes use relative values.
                '''
                bbox[0] *= 1. / img_data['width'] # x
                bbox[1] *= 1. / img_data['height']  # y
                bbox[2] = bbox[0] + bbox[2] * (1. / img_data['width'])  # width
                bbox[3] = bbox[1] + bbox[3] * (1. / img_data['height'])  # height
                # assign box format ([xmin,ymin,xmax,ymax] + [one_hot(80)])
                classes = np.zeros(80)
                classes[catsToIds[ann['category_id']]] = 1
                bboxList.append(np.concatenate((bbox, classes)))
            Y.append(bboxer.assign_boxes(np.array(bboxList)))
            n += 1
        Y = np.array(Y)
        yield X, Y


g = generator(batch_size=2)

model.fit_generator(generator=g, samples_per_epoch=100, nb_val_samples=10, validation_data=g, nb_epoch=42)
