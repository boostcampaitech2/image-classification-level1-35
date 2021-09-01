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
    def __init__(self, img_list, label_list, transform):
        self.X = img_list
        self.y = label_list
        self.transform = transform
    
    # 길이 반환
    def __len__(self):
        return len(self.y)
    
    # 데이터 반환
    def __getitem__(self, index):
        img = np.array(Image.open(self.X[index]))
        
        # 이미지는 전처리 적용해서 반환
        if self.transform:
            img = (self.transform(image=img))['image']

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

def new_train_dataset(train_path, img_path):

    raw = pd.read_csv(train_path)

    #사람별 폴더의 파일 7개 경로 가져오기
    path = []
    route = []
    
    for v in raw['path']:
        path.append(os.listdir(img_path + '/' + v))


    #새로운 데이터셋을 위한 mask, path(사진별 파일)의 feature생성하기
    mask = []
    pic_path = []

    for i_d in path:
        for pic in i_d:
            if 'aug' in pic:
                continue
            if pic[0] == 'm':
                mask.append('wear')
                pic_path.append(pic)
            elif pic[0] == 'i':
                mask.append('incorrect')
                pic_path.append(pic)
            elif pic[0] == 'n':
                mask.append('not wear')
                pic_path.append(pic)
            else:
                pass


    #기존의 데이터셋에 있는 id, age, gender, path(사람별 폴더) 개수 맞추기
    person_path = []
    for i in raw['path']:
        for j in range(7):
            person_path.append(i)

    id = []
    for i in raw['id']:
        for j in range(7):
            id.append(i)


    gender = []
    for i in raw['gender']:
        for j in range(7):
            gender.append(i)

    age = []
    for i in raw['age']:
        for j in range(7):
            age.append(i)

    #사람별 + 사진별 경로 합쳐서 최종 path 생성하기
    final_path = []
    for i in range(18900):
        final_path.append(os.path.join(img_path ,person_path[i] + '/' + pic_path[i])) 

    #새로운 데이터셋 생성하기
    df = pd.DataFrame({'id': id, 
                    'gender': gender, 
                    'age': age, 
                    'mask': mask, 
                    'path': final_path})

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
    # 18 클래스 생성(라벨링)
    elif model_type == 'All':
        # mask=wear
        df.loc[(df['gender']=='male')&(df['age'] < 30)&(df['mask']=='wear'), 'class'] = 0
        df.loc[(df['gender']=='male')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='wear'), 'class'] = 1
        df.loc[(df['gender']=='male')&(df['age'] >= 60)&(df['mask']=='wear'), 'class'] = 2            
        df.loc[(df['gender']=='female')&(df['age'] < 30)&(df['mask']=='wear'), 'class'] = 3
        df.loc[(df['gender']=='female')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='wear'), 'class'] = 4
        df.loc[(df['gender']=='female')&(df['age'] >= 60)&(df['mask']=='wear'), 'class'] = 5

        #mask=incorrect
        df.loc[(df['gender']=='male')&(df['age'] < 30)&(df['mask']=='incorrect'), 'class'] = 6
        df.loc[(df['gender']=='male')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='incorrect'), 'class'] = 7
        df.loc[(df['gender']=='male')&(df['age'] >= 60)&(df['mask']=='incorrect'), 'class'] = 8
        df.loc[(df['gender']=='female')&(df['age'] < 30)&(df['mask']=='incorrect'), 'class'] = 9
        df.loc[(df['gender']=='female')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='incorrect'), 'class'] = 10
        df.loc[(df['gender']=='female')&(df['age'] >= 60)&(df['mask']=='incorrect'), 'class'] = 11

        #mask=normal
        df.loc[(df['gender']=='male')&(df['age'] < 30)&(df['mask']=='not wear'), 'class'] = 12
        df.loc[(df['gender']=='male')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='not wear'), 'class'] = 13
        df.loc[(df['gender']=='male')&(df['age'] >= 60)&(df['mask']=='not wear'), 'class'] = 14            
        df.loc[(df['gender']=='female')&(df['age'] < 30)&(df['mask']=='not wear'), 'class'] = 15
        df.loc[(df['gender']=='female')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='not wear'), 'class'] = 16
        df.loc[(df['gender']=='female')&(df['age'] >= 60)&(df['mask']=='not wear'), 'class'] = 17
    else:
        print('Wrong Prediction Type!!')
        exit(1)
    df = df.astype({'class':int})
    return df

def make_train_list(df, config, valid_ids):
    if config.prediction_type in ['Age','Gender']:
        if config.learning_type == 'None':
            train_list, train_label = df[(~df['id'].isin(valid_ids)) & (df['mask']=='not wear')]['path'], df[(~df['id'].isin(valid_ids)) & (df['mask']=='not wear')]['class']
            valid_list, valid_label = df[(df['id'].isin(valid_ids)) & (df['mask']=='not wear')]['path'], df[(df['id'].isin(valid_ids)) & (df['mask']=='not wear')]['class']
        elif config.learning_type == 'Mask':
            train_list, train_label = df[(~df['id'].isin(valid_ids)) & ((df['mask']=='wear') | (df['mask']=='incorrect'))]['path'], df[(~df['id'].isin(valid_ids)) & ((df['mask']=='wear') | (df['mask']=='incorrect'))]['class']
            valid_list, valid_label = df[df['id'].isin(valid_ids)]['path'], df[df['id'].isin(valid_ids)]['class']
        elif config.learning_type == 'All':
            train_list, train_label = df[(~df['id'].isin(valid_ids))]['path'], df[(~df['id'].isin(valid_ids))]['class']
            valid_list, valid_label = df[df['id'].isin(valid_ids)]['path'], df[df['id'].isin(valid_ids)]['class']
        else:
            print("Wrong learning type!!")
            exit(1)
    else:
        train_list, train_label = df[(~df['id'].isin(valid_ids))]['path'], df[(~df['id'].isin(valid_ids))]['class']
        valid_list, valid_label = df[df['id'].isin(valid_ids)]['path'], df[df['id'].isin(valid_ids)]['class']

    return train_list, train_label, valid_list, valid_label

def make_fold(fold_num, df):
    folds = []
    df2 = df
    num_of_person = len(pd.unique(df['id']))
    # ver1
    # for i in range(fold_num):
    #     v = df2.groupby('id')['id'].sample(n=1).sample(n=fold_size, random_state=42, replace=False)
    #     df2 = df2[~df2['id'].isin(v)]
    #     folds.append(v)
    # del df2
    
    # ver2
    # for i in range(fold_num):
    #     fold_ratio = np.around(540 / len(set(df2['id'])),1)
    #     train, test = train_test_apart_stratify(df2, group="id", stratify="class", force=True, test_size=fold_ratio, random_state = 42)
    #     df2 = df2[~df2['id'].isin(pd.unique(test['id']))]
    #     folds.append(test['id'])
    # del df2

    # return folds
    # cnt = 0
    for i in range(fold_num):
        # fold_ratio = [0.2, 0.3, 0.4, 0.5, 0.6]
        fold_ratio = np.around(540 / len(set(df2['id'])),1)
        train, test = train_test_apart_stratify(df2, group="id", stratify="class", force=True, test_size=fold_ratio, random_state = 42)
        df2 = df2[~df2['id'].isin(pd.unique(test['id']))]
        folds.append(test['id'])
        cnt += 1
    del df2
    return folds

    # for i in range(fold_num):
    #     fold_ratio = np.around(540 / len(set(df2['id'])), 1)
    #     train, test = train_test_apart_stratify(df2, group="id", stratify="class", force=True, test_size=fold_ratio, random_state = 42)
    #     df2 = df2[~df2['id'].isin(pd.unique(test['id']))]
    #     folds.append(test['id'])
    # del df2
    # return folds



def read_age_data():
    path = '/opt/ml/input/data/train/add_Age2'
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
    elif 30 <= age <60:
        return 1
    else:
        return 2


# def read_age_data():
#     path = '../../input/data/train/Age/'
#     img_dict = {'id':[], 'age':[], 'path':[], 'class':[]}
#     for img_path in os.listdir(path):
#         if img_path == '.ipynb_checkpoints':
#             continue
#         try:
#             age, id = img_path.split('(')
#         except:
#             print(img_path)
#             exit(1)
#         img_dict['id'].append(id[:-5])
#         img_dict['age'].append(int(age))
#         img_dict['path'].append(os.path.join(path, img_path))
#         img_dict['class'].append(get_label_added_data_for_age(int(age)))

#     return pd.DataFrame(img_dict)

# def get_label_added_data_for_age(age):
#     if age < 30:
#         return 0
#     elif age < 60:
#         return 1
#     else:
#         return 2


# 이미지 전체 경로 생성
# train.csv의 path 컬럼 이용
# def path_maker(csv_paths, image_path, load_augmentation):
#     path_list = []  # 모든 이미지 경로들 저장할 리스트
#     label_list = [] # 모든 이미지 라벨들 저장할 리스트
    
#     # train.csv 파일 읽어오기
#     img_dir = pd.read_csv(csv_paths)#"../input/data/train/train.csv")
    
#     # path 컬럼만 추출
#     img_dir_pathes = img_dir['path']
    
#     # image_path = ../input/data/train/images" + img_dir['path']의 이미지 폴더 이름을 합치기
#     # 결과예시: ../input/data/train/images/[사람별 고유 폴더명]
#     img_dir_patheses = [os.path.join(image_path, img_d) for img_d in img_dir_pathes]
   
#     # 풀 경로 만들기 위한 반복문/mask1
#     # 결과예시: path_list => ../input/data/train/images/[사람별 고유 폴더명]/mask1.jpg
#     #        label_list 에는 라벨들 저장
#     for f_name in img_dir_patheses:
#         for img_name in os.listdir(f_name):
#             if img_name[0] == '.': # .으로 시작하는 바이너리 파일 skip
#                 continue
#             if not load_augmentation and 'aug' in img_name:
#                 continue
#             # 예시: ../input/data/train/images/[사람별 고유 폴더명]와 'mask1.jpg' 결합
#             maked_path = os.path.join(f_name, img_name)
            
#             # GetLabel 함수로 라벨변환
#             label = get_label(maked_path)
            
#             # 리스트에 넣어주기
#             path_list.append(maked_path)
#             label_list.append(label)
#     return np.array(path_list), np.array(label_list)

# def get_label(path):
#         # 전체경로에서 마지막 부분 가져 오기 => [사람별 고유 폴더명], mask1.jpg (이부분)
#         items = path.split('/')[-2:]
        
#         # [사람별 고유 폴더명] 정보 분해하기
#         number, sex, race, age = items[0].split('_')
        
#         # mask1 부분 가져오기
#         mask_status = items[1].split('.')[0]
        
#         # age 단일 라벨로 수정
#         if int(age) < 30:
#             return 0
#         elif int(age) < 60:
#             return 1
#         else:
#             return 2

#         # class 조건에 따라 분류
#         # mask먼저 분류하면 incorrect_mask도 포함이 되서 incorrect 먼저 분류
#         # if 'incorrect' in mask_status:
#         #     if sex == 'male':
#         #         if int(age) < 30:
#         #             return 6
#         #         elif int(age) < 60:
#         #             return 7
#         #         else:
#         #             return 8
#         #     else:
#         #         if int(age) < 30:
#         #             return 9
#         #         elif int(age) < 60:
#         #             return 10
#         #         else:
#         #             return 11
#         # elif 'mask' in mask_status:
#         #     if sex == 'male':
#         #         if int(age) < 30:
#         #             return 0
#         #         elif int(age) < 60:
#         #             return 1
#         #         else:
#         #             return 2
#         #     else:
#         #         if int(age) < 30:
#         #             return 3
#         #         elif int(age) < 60:
#         #             return 4
#         #         else:
#         #             return 5
#         # else:
#         #     if sex == 'male':
#         #         if int(age) < 30:
#         #             return 12
#         #         elif int(age) < 60:
#         #             return 13
#         #         else:
#         #             return 14
#         #     else:
#         #         if int(age) < 30:
#         #             return 15
#         #         elif int(age) < 60:
#         #             return 16
#         #         else:
#         #             return 17
