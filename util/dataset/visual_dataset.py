import json
import os

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import pandas as pd



class contrastive_pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):    
        
        ann = self.ann[index]

        img_name= ann['url'].split('/')[-1]
        img_path = os.path.join('data/images_train/', img_name)


        image = Image.open(img_path).convert('RGB')
        image1 = self.transform(image)
        image2 = self.transform(image)
                
        return image1, image2


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform

        print('loading json file done!')

    def __len__(self):
        return len(self.ann)


    def __getitem__(self, index):

        ann = self.ann[index]

        labels_inci = ann['incidents_pos_num'][0]
        labels_plac = ann['places_pos_num'][0]

        img_name = ann['image']
        img_path = os.path.join('../../datasets/incidentonem/images/images_train/', img_name)
        # img_path = os.path.join('../datasets/incidentonem_data2img/images/images_train/images_train/', img_name)

        image = Image.open(img_path).convert('RGB')
        image1 = self.transform(image)

        return image1, labels_inci, labels_plac


class binary_incidentonem(Dataset):
    def __init__(self, ann_file, transform, task, category):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.task = task
        self.category = category
        print('loading json file done!')

    def __len__(self):
        return len(self.ann)


    def __getitem__(self, index):

        ann = self.ann[index]

        label = ann[self.task][self.category]

        img_name = ann['image']
        img_path = os.path.join('../datasets/incidentonem/images/images_train/', img_name)


        image = Image.open(img_path).convert('RGB')
        image1 = self.transform(image)

        return image1, label




class crisis_image_benchmarks(Dataset):
    def __init__(self, ann_file, transform):
        self.df = pd.read_csv(ann_file, sep=',')
        self.transform = transform

        print(f'loading csv {ann_file} file done!')

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        ann = self.df.iloc[index]

        label = ann['class_label_num']

        img_name = ann['image_path']
        img_path = os.path.join('../datasets/crisis_vision_benchmarks', img_name)

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label