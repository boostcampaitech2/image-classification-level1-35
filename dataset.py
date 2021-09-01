import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from albumentations import *
import matplotlib.pyplot as plt
from pandas_streaming.df import train_test_apart_stratify
import random

# Data augmentation을 위한 클래스
class Preprocessing():
    def __init__(self, pathes, labels, targets, aug_num):
        self.pathes = pathes
        self.y = labels
        self.targets = targets
        self.aug_num = aug_num

    # AutoAugmentPolicy = IMAGENET
    def augmentation(self):
        policy = T.AutoAugmentPolicy.IMAGENET
        augmenter = T.AutoAugment(policy)
        
        for idx, path in enumerate(self.pathes):
            # 정해진 클래스가 아니면 augmentation 실행 X
            if self.y[idx] not in self.targets:
                continue
            img = Image.open(path)
            for _ in range(self.aug_num):
                processed_img = augmenter(img)
                file_name = path.split('/')[-1].split('.')[0]
                new_path = path.replace(file_name, file_name + "_aug_" + str(_))

                # 이미지 저장
                processed_img.save(new_path)

# Dataset 클래스 정의
class TrainDataset(Dataset):
    # img_list = 훈련할 이미지 경로들
    # label_list = 위 이미지에 맞는 라벨들
    # transform = 전처리 정의
    def __init__(self, img_list, label_list, transform, config):
        self.X = img_list
        self.y = label_list
        self.transform = transform
        self.config = config
    
    # 길이 반환
    def __len__(self):
        return len(self.y)
    
    # 데이터 반환
    def __getitem__(self, index):
        img = np.array(Image.open(self.X[index]))
        
        # 이미지는 전처리 적용해서 반환
        if self.transform:
            img = (self.transform(image=img))['image']

        if self.config.mode == 'Regression':
            return img.float(), torch.tensor([self.y[index]]).float()
        else:
            return img.float(), self.y[index]

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image.float()

    def __len__(self):
        return len(self.img_paths)

def new_train_dataset(train_path, img_path, config):

    raw = pd.read_csv(train_path)

    #사람별 폴더의 파일 7개 경로 가져오기
    path = []
    new_dict = {
        'id':[],
        'age' : [],
        'gender' :[],
        'mask' :[],
        'path' :[]
    }
    for raw_idx, v in enumerate(raw['path']):
        person_path = img_path + '/' + v
        number, gender, race, age = v.split('_')
        for imgp in os.listdir(person_path):
            if not config.load_augmentation and 'aug' in imgp:
                continue
            if imgp[0] == 'm':
                new_dict['id'].append(raw['id'].iloc[raw_idx])
                new_dict['mask'].append('wear')
                new_dict['gender'].append(gender)
                new_dict['age'].append(float(age))
                new_dict['path'].append(os.path.join(person_path, imgp))
            elif imgp[0] == 'i':
                new_dict['id'].append(raw['id'].iloc[raw_idx])
                new_dict['mask'].append('incorrect')
                new_dict['gender'].append(gender)
                new_dict['age'].append(float(age))
                new_dict['path'].append(os.path.join(person_path, imgp))
            elif imgp[0] == 'n':
                new_dict['id'].append(raw['id'].iloc[raw_idx])
                new_dict['mask'].append('not wear')
                new_dict['gender'].append(gender)
                new_dict['age'].append(float(age))
                new_dict['path'].append(os.path.join(person_path, imgp))
            else:
                pass

    df = pd.DataFrame(new_dict)
    return df

def get_label(df, model_type):
    if model_type == 'Mask':
        df.loc[(df['mask']=='wear'), 'class'] = 0
        df.loc[(df['mask']=='not wear'), 'class'] = 1
        df.loc[(df['mask']=='incorrect'), 'class'] = 2
    elif model_type == 'Age':
        df.loc[(df['age'] < 30), 'class'] = 0
        df.loc[(df['age'] >= 30)&(df['age']< 60), 'class'] = 1
        df.loc[(df['age'] >= 60), 'class'] = 2            
    elif model_type == 'Gender':
        df.loc[(df['gender']=='male'), 'class'] = 0
        df.loc[(df['gender']=='female'), 'class'] = 1
    else:
        print('Wrong Prediction Type!!')
        exit(1)
    # 18 클래스 생성(라벨링)
    #mask=wear
    # df.loc[(df['gender']=='male')&(df['age'] < 30)&(df['mask']=='wear'), 'class'] = 0
    # df.loc[(df['gender']=='male')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='wear'), 'class'] = 1
    # df.loc[(df['gender']=='male')&(df['age'] >= 60)&(df['mask']=='wear'), 'class'] = 2            
    # df.loc[(df['gender']=='female')&(df['age'] < 30)&(df['mask']=='wear'), 'class'] = 3
    # df.loc[(df['gender']=='female')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='wear'), 'class'] = 4
    # df.loc[(df['gender']=='female')&(df['age'] >= 60)&(df['mask']=='wear'), 'class'] = 5

    # #mask=incorrect
    # df.loc[(df['gender']=='male')&(df['age'] < 30)&(df['mask']=='incorrect'), 'class'] = 6
    # df.loc[(df['gender']=='male')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='incorrect'), 'class'] = 7
    # df.loc[(df['gender']=='male')&(df['age'] >= 60)&(df['mask']=='incorrect'), 'class'] = 8
    # df.loc[(df['gender']=='female')&(df['age'] < 30)&(df['mask']=='incorrect'), 'class'] = 9
    # df.loc[(df['gender']=='female')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='incorrect'), 'class'] = 10
    # df.loc[(df['gender']=='female')&(df['age'] >= 60)&(df['mask']=='incorrect'), 'class'] = 11

    # #mask=normal
    # df.loc[(df['gender']=='male')&(df['age'] < 30)&(df['mask']=='not wear'), 'class'] = 12
    # df.loc[(df['gender']=='male')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='not wear'), 'class'] = 13
    # df.loc[(df['gender']=='male')&(df['age'] >= 60)&(df['mask']=='not wear'), 'class'] = 14            
    # df.loc[(df['gender']=='female')&(df['age'] < 30)&(df['mask']=='not wear'), 'class'] = 15
    # df.loc[(df['gender']=='female')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='not wear'), 'class'] = 16
    # df.loc[(df['gender']=='female')&(df['age'] >= 60)&(df['mask']=='not wear'), 'class'] = 17
    df = df.astype({'class':int})
    return df

def make_train_list(df, config, valid_ids):
    target = 'class'
    if config.prediction_type == 'Age' and config.mode == 'Regression' :
        target = 'age'
    if config.prediction_type  in ['Age', 'Gender']:
        if config.learning_type == 'None':
            train_list, train_label = df[(~df['id'].isin(valid_ids)) & (df['mask']=='not wear')]['path'], df[(~df['id'].isin(valid_ids)) & (df['mask']=='not wear')][target]
            valid_list, valid_label = df[(df['id'].isin(valid_ids)) & (df['mask']=='not wear') & (~df['path'].str.contains('aug'))]['path'], df[(df['id'].isin(valid_ids)) & (df['mask']=='not wear') & (~df['path'].str.contains('aug'))][target]
        elif config.learning_type == 'Mask':
            train_list, train_label = df[(~df['id'].isin(valid_ids)) & ((df['mask']=='wear') | (df['mask']=='incorrect'))]['path'], df[(~df['id'].isin(valid_ids)) & ((df['mask']=='wear') | (df['mask']=='incorrect'))][target]
            valid_list, valid_label = df[df['id'].isin(valid_ids)& (~df['path'].str.contains('aug'))]['path'], df[df['id'].isin(valid_ids)& (~df['path'].str.contains('aug'))][target]
        elif config.learning_type == 'All':
            train_list, train_label = df[(~df['id'].isin(valid_ids))]['path'], df[(~df['id'].isin(valid_ids))][target]
            valid_list, valid_label = df[df['id'].isin(valid_ids)& (~df['path'].str.contains('aug'))]['path'], df[df['id'].isin(valid_ids)& (~df['path'].str.contains('aug'))][target]
        else:
            print("Wrong learning type!!")
            exit(1)
    else:
        train_list, train_label = df[(~df['id'].isin(valid_ids))]['path'], df[(~df['id'].isin(valid_ids))][target]
        valid_list, valid_label = df[df['id'].isin(valid_ids) & (~df['path'].str.contains('aug'))]['path'], df[df['id'].isin(valid_ids)& (~df['path'].str.contains('aug'))][target]
    
    return train_list, train_label, valid_list, valid_label

def make_fold(fold_num, df):
    folds = []
    df2 = df
    num_of_person = len(pd.unique(df['id']))
    fold_size = int(num_of_person / fold_num)
    # ver1
    # for i in range(fold_num):
    #     v = df2.groupby('id')['id'].sample(n=1).sample(n=fold_size, random_state=42, replace=False)
    #     df2 = df2[~df2['id'].isin(v)]
    #     folds.append(v)
    # del df2
    
    # ver2
    # aug로 인한 균형 맞춰진것 영향 없애기
    df2 = df[(~df['path'].str.contains('aug'))]
    for i in range(fold_num):
        train, test = train_test_apart_stratify(df2, group="id", stratify="class", force=True, test_size=0.2, random_state = 42)
        df2 = df2[~df2['id'].isin(pd.unique(test['id']))]
        folds.append(test['id'])
    del df2

    return folds

def read_age_data():
    path = '../../input/data/train/Age/'
    img_dict = {'id':[], 'age':[], 'path':[], 'class':[]}
    for img_path in os.listdir(path):
        if img_path == '.ipynb_checkpoints':
            continue
        try:
            age, id = img_path.split('(')
        except:
            print(img_path)
            exit(1)
        img_dict['id'].append(id[:-5])
        img_dict['age'].append(int(age))
        img_dict['path'].append(os.path.join(path, img_path))
        img_dict['class'].append(get_label_added_data_for_age(int(age)))

    return pd.DataFrame(img_dict)

def get_label_added_data_for_age(age):
    if age < 30:
        return 0
    elif age < 60:
        return 1
    else:
        return 2
        
class SiameseNetworkDataset(Dataset):
    
    def __init__(self, img_list, label_list, transform):
        self.img_list = img_list    
        self.label_list = label_list    
        self.transform = transform
        
    def __getitem__(self, index):
        target_image_idx = index
        
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                compare_image_idx = np.random.choice(len(self.label_list), 1)
                if self.label_list[index] == self.label_list[compare_image_idx]:
                    break
        else:
            while True:
                #keep looping till a different class image is found  
                compare_image_idx = np.random.choice(len(self.label_list), 1)
                if self.label_list[target_image_idx] == self.label_list[compare_image_idx]:
                    break

        target_image = Image.open(self.img_list[target_image_idx])
        compare_image = Image.open(self.img_list[compare_image_idx])
        target_image = target_image.convert("L")
        compare_image = compare_image.convert("L")
        
        # if self.should_invert:
        #     target_image = PIL.ImageOps.invert(target_image)
        #     compare_image = PIL.ImageOps.invert(compare_image)
        if self.transform:
            target_image = self.transform(image=np.array(target_image))['image']
            compare_image = self.transform(image=np.array(compare_image))['image']
          
        return target_image.float(), compare_image.float() , torch.from_numpy(np.array([int(self.label_list[target_image_idx]!=self.label_list[compare_image_idx])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
