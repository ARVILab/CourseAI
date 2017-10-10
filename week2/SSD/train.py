from ssd import SSD300
from ssd_utils import BBoxUtility
from ssd_training import MultiboxLoss
from pycocotools.coco import COCO
import numpy as np
import cv2
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


print('ololo')
def generator(batch_size=4):
    imgList = cocodata.imgs.keys()
    imgcount =len(imgList)
    n=0
    while True:
        X = np.zeros((batch_size, 300,300,3), dtype=np.float)
        Y = []
        for i in range(0, batch_size):
            n = n%imgcount
            imgData = cocodata.imgs[imgList[n]]
            imgPath = cocoDir + imgData['file_name']
            img = cv2.imread(imgPath)
            X[i] = cv2.resize(img, (300,300), interpolation=cv2.INTER_CUBIC).astype(np.float) / 127.5 - 1.
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
                '''
                bbox[0] *= 300. / img_data['width']  # x
                bbox[1] *= 300. / img_data['height']  # y
                bbox[3] *= 300. / img_data['width']  # width
                bbox[4] *= 300. / img_data['height']  # height
                classes = np.zeros(80)
                classes[catsToIds[ann['category_id']]] = 1
                bboxList.append(np.concatenate((bbox,classes)))
            Y.append(bboxer.assign_boxes(np.array(bboxList)))
            n+=1
        Y = np.array(Y)
        yield X, Y

g = generator(batch_size=2)

model.fit_generator(generator=g, samples_per_epoch=100, nb_val_samples=10, validation_data=g, nb_epoch=42)
