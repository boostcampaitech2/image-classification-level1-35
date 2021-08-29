import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from albumentations import *
import matplotlib.pyplot as plt

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

# 이미지 전체 경로 생성
# train.csv의 path 컬럼 이용
def path_maker(csv_paths, image_path, load_augmentation):
    path_list = []  # 모든 이미지 경로들 저장할 리스트
    label_list = [] # 모든 이미지 라벨들 저장할 리스트
    
    # train.csv 파일 읽어오기
    img_dir = pd.read_csv(csv_paths)#"../input/data/train/train.csv")
    
    # path 컬럼만 추출
    img_dir_pathes = img_dir['path']
    
    # image_path = ../input/data/train/images" + img_dir['path']의 이미지 폴더 이름을 합치기
    # 결과예시: ../input/data/train/images/[사람별 고유 폴더명]
    img_dir_patheses = [os.path.join(image_path, img_d) for img_d in img_dir_pathes]
   
    # 풀 경로 만들기 위한 반복문/mask1
    # 결과예시: path_list => ../input/data/train/images/[사람별 고유 폴더명]/mask1.jpg
    #        label_list 에는 라벨들 저장
    for f_name in img_dir_patheses:
        for img_name in os.listdir(f_name):
            if img_name[0] == '.': # .으로 시작하는 바이너리 파일 skip
                continue
            if not load_augmentation and 'aug' in img_name:
                continue
            # 예시: ../input/data/train/images/[사람별 고유 폴더명]와 'mask1.jpg' 결합
            maked_path = os.path.join(f_name, img_name)
            
            # GetLabel 함수로 라벨변환
            label = get_label(maked_path)
            
            # 리스트에 넣어주기
            path_list.append(maked_path)
            label_list.append(label)
    return np.array(path_list), np.array(label_list)

def get_label(path):
        # 전체경로에서 마지막 부분 가져 오기 => [사람별 고유 폴더명], mask1.jpg (이부분)
        items = path.split('/')[-2:]
        
        # [사람별 고유 폴더명] 정보 분해하기
        number, sex, race, age = items[0].split('_')
        
        # mask1 부분 가져오기
        mask_status = items[1].split('.')[0]

        # class 조건에 따라 분류
        # mask먼저 분류하면 incorrect_mask도 포함이 되서 incorrect 먼저 분류
        if 'incorrect' in mask_status:
            if sex == 'male':
                if int(age) < 30:
                    return 6
                elif int(age) < 60:
                    return 7
                else:
                    return 8
            else:
                if int(age) < 30:
                    return 9
                elif int(age) < 60:
                    return 10
                else:
                    return 11
        elif 'mask' in mask_status:
            if sex == 'male':
                if int(age) < 30:
                    return 0
                elif int(age) < 60:
                    return 1
                else:
                    return 2
            else:
                if int(age) < 30:
                    return 3
                elif int(age) < 60:
                    return 4
                else:
                    return 5
        else:
            if sex == 'male':
                if int(age) < 30:
                    return 12
                elif int(age) < 60:
                    return 13
                else:
                    return 14
            else:
                if int(age) < 30:
                    return 15
                elif int(age) < 60:
                    return 16
                else:
                    return 17
        
# Dataset 클래스 정의
class TrainDataset_v2(Dataset):
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