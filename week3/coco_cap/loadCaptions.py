import pickle as pkl
import numpy as np

captions = pkl.load(open('/DATA/CourseAI/datasets/coco/captions_df.pkl','rb'))
print('trololo')

allCapsData = []

for i in range(0,len(captions.values)):
    data = captions.values[i]
    for j in range(0,5):
        allCapsData.append([data[-1].replace('\n','').replace('\n',''), data[j].lower().replace('.','')])
data = np.array(allCapsData)
np.save('AllTrainCaps.npy', data)