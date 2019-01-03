import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod


class CocoDataset(data.Dataset):

    def __init__(self, root, json, vocab, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        # if ids provided by get_paths, use split-specific ids
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        image, caption, img_id = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        target = self.encode_token(caption)
        target = torch.Tensor(target)
        return image, target, index, img_id

    def encode_token(self, caption):
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        target = [ self.vocab(token) for token in tokens ]
        return target

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')

        return image, caption, img_id

    def __len__(self):
        return len(self.ids)


def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        for name in ['train', 'val']:
            roots[name] = {
                'img': os.path.join(imgdir, '{}2014'.format(name)),
                'cap': os.path.join(capdir, 'captions_{}2014.json'.format(name))
            }
        roots['test'] = roots['val']
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']

    return roots, ids


def get_loaders(data_name, vocab, crop_size, batch_size, num_workers, opt):
    dpath = os.path.join(opt.data_path, data_name) # /DATA/cvpr19/coco
    # Build Dataset Loader
    roots, ids = get_paths(dpath, data_name, opt.use_restval)

    loaders = {}
    for split in ['train', 'val']:
        # COCO custom dataset
        transform = get_transform(data_name, split, crop_size)
        dataset = CocoDataset(root=roots[split]['img'],
                              json=roots[split]['cap'],
                              vocab=vocab,
                              transform=transform, ids=ids[split])
        loaders[split] = data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=(split=='train'),
                                         pin_memory=True,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn)

    return loaders['train'], loaders['val']


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.

    """
    # Sort a data list by caption length

    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, index, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, index


def get_transform(data_name, split_name, crop_size):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    # Build Dataset Loader
    roots, ids = get_paths(dpath, data_name, opt.use_restval)

    transform = get_transform(data_name, split_name, crop_size)
    test_loader = get_loader_single(opt.data_name, split_name,
                                    roots[split_name]['img'],
                                    roots[split_name]['cap'],
                                    vocab, transform, ids=ids[split_name],
                                    batch_size=batch_size, shuffle=False,
                                    num_workers=workers,
                                    collate_fn=collate_fn)

    return test_loader

