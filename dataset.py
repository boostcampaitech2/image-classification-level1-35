import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import func


#데이터셋 불러오기

#train_path = '/opt/ml/input/data/train/train.csv'
#img_path = '/opt/ml/input/data/train/images'

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
        final_path.append(person_path[i] + '/' + pic_path[i]) 

    '''
    #png, jpeg파일 jpg로 변환하기
    train_mask_path = []
    for i in raw['path']:
        train_mask_path.append('/opt/ml/input/data/train/images' + '/' + i)

    not_jpg_file = []
    for pic_path in train_mask_path:
        t = pic_path.split('.')[-1]
        if len(t) > 4:
            continue
        if t != 'jpg':
            not_jpg_file.append(pic_path)

    for imagedir in not_jpg_file:
        im = Image.open(imagedir)
        im.save('.' + imagedir.split('.')[1] + '.jpg')

    #기존의 png, jpeg파일 삭제하기
    for i in not_jpg_file:
        os.remove(i)

    '''


    #새로운 데이터셋 생성하기
    df = pd.DataFrame({'id': id, 
                    'gender': gender, 
                    'age': age, 
                    'mask': mask, 
                    'path': final_path})

    return df


def get_label(new_train_dataset):
    #클래스 생성(라벨링)
    #mask=wear
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


    df = df.astype({'class':int})

    return df


# img_path = '/opt/ml/input/data/train/images/'
class MaskDataset(Dataset):
    def __init__(self, df, img_path, transform):
        self.df = df
        self.img_path = img_path
        self.X = self.img_path + self.df['path']
        self.y = self.df['class']
        self.classes = set(self.y)
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        y = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y)