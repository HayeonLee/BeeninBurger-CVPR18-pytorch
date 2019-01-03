import json
import numpy as np
jf = json.load(open('/DATA/cvpr19/coco/annotations/dataset.json', 'rb'))['images']

train = []
restval = []
val = []
test = []

def get_ids(data):
    tmp = []
    for j in range(5):
      tmp.append(data['sentids'][j])
    return tmp

for i in range(len(jf)):
    if jf[i]['split'] in ['train']:
      train += get_ids(jf[i])
    elif jf[i]['split'] in ['restval']:
      restval += get_ids(jf[i])
    elif jf[i]['split'] in ['val']:
      val += get_ids(jf[i])
    elif jf[i]['split'] in ['test']:
      test += get_ids(jf[i])

ids = {'train': train,
       'restval': restval,
       'test': test,
       'dev': val}

for name in ['train', 'restval', 'test', 'dev']:
    np.save('/DATA/cvpr19/coco/annotations/coco_{}_ids.npy'.format(name), np.array(ids[name]))
