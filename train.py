import os
import sys, getopt
import pandas as pd
from pandas.core.arrays.sparse import dtype
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch.utils.data import sampler
from albumentations import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models  as cvmodels
from torchsummary import summary

from model import *
from dataset import *
from utill import * 
from Loss import *
import random
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

import wandb

def train(train_loader, valid_loader, class_weigth, fold_index, config):
    # 모델 생성
    print("Model Generation...")
    #model = MyModel(num_classes=18).to(device)
    # model = torch.load("../models/Resnet101_dense.pt").to(device)
    # model = vision_transformer(
    #     in_channel = 3, 
    #     img_size = 256,
    #     patch_size = 8,
    #     emb_dim = 16*16,
    #     num_heads = 2,
    #     n_enc_layers = 3,
    #     forward_dim= 3,
    #     dropout_ratio = 0.1,
    #     n_classes = 18).to(device)
    #model = Efficientnet(num_classes=18).to(config.device)
    model = get_model(config, num_classes=3)
    wandb.watch(model)
    #model = SWSLResnext50(num_classes = 18).to(device)
    # Backbone freezing
    # for p in model.pretrained.parameters(): # Resnet part
    #     p.requires_grad = False
    # for p in model.pretrained.fc.parameters(): # Resnet part
    #     p.requires_grad = True

    # 모델 정보 출력
    print(model)
    #summary(model, input_size=(3, 512, 384), device=device)

    # 학습
    if config.loss == 'CrossEntropy':
        loss_func1 = torch.nn.CrossEntropyLoss()
    elif config.loss == 'Crossentropy_foscal':
        loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weigth).to(config.device, dtype=torch.float))
        loss_func2 = FocalLoss()
    elif config.loss == 'CrossEntropy_weighted':
        loss_func1 = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weigth).to(config.device, dtype=torch.float))
    elif config.loss == 'Foscal':
        loss_func1 = FocalLoss()
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)
    elif config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)

    if config.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 10,
            eta_min = 0  
        )
    # lrscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
    #                     lr_lambda=lambda epoch: 0.95 ** epoch,
    #                     last_epoch=-1,
    #                     verbose=False)
   
    best_metric = 0
    best_model_dict = None
    early_stopping_count = 0
    # result = {
    #     'train_loss':[],
    #     'train_f1':[],
    #     'valid_loss':[],
    #     'valid_acc':[],
    #     'valid_f1':[]
    #     }
    print("-"*10, "Training", "-"*10)
    for e in range(1, config.epoches + 1):
        batch_corss_loss = 0
        batch_foscal_loss = 0
        batch_loss = 0
        batch_f1_pred = []
        batch_f1_target = []
        batch_f1 = 0
        # train
        for tr_idx, (X, y) in enumerate(tqdm(train_loader)):
            x = X.to(config.device)
            y = y.to(config.device)
            optimizer.zero_grad()

            pred = model.forward(x)
            # cross_loss = loss_func1(pred, y)
            # foscal_loss = loss_func2(pred, y)
            if config.loss == 'CrossEntropy' or config.loss == 'CrossEntropy_weighted' or config.loss == 'Foscal':
                cross_loss = loss_func1(pred, y)
                loss = cross_loss
                
                batch_corss_loss += cross_loss.cpu().data
            elif config.loss == 'Crossentropy_foscal':
                cross_loss = loss_func1(pred, y)
                foscal_loss = loss_func2(pred, y)

                batch_corss_loss += cross_loss.cpu().data
                batch_foscal_loss += foscal_loss.cpu().data
                loss = cross_loss * config.loss1_weight + foscal_loss * config.loss2_weight
            loss.backward()
            optimizer.step()
  
            batch_loss += loss.cpu().data
            batch_f1_pred.extend(torch.argmax(pred.cpu().data, dim=1))
            batch_f1_target.extend(y.cpu().data)
        
        # validation
        model.eval()
        running_acc = 0
        running_loss = 0
        running_f1_pred = []
        running_f1_target = []
        examples = []
        for te_idx, (X, y) in enumerate(tqdm(valid_loader)):
            X = X.to(config.device)
            y = y.to(config.device)

            with torch.set_grad_enabled(False):
                pred = model(X)
                if config.loss == 'CrossEntropy' or config.loss == 'CrossEntropy_weighted' or config.loss == 'Foscal':
                    loss = loss_func1(pred, y)
                else:
                    loss1 = loss_func1(pred, y)
                    loss2 = loss_func2(pred, y)
                    loss = loss1 * config.loss1_weight + loss2 * config.loss2_weight
                running_acc += accuracy_score(torch.argmax(pred.cpu().data, dim=1), y.cpu().data)
                running_f1_pred.extend(torch.argmax(pred.cpu().data, dim=1))
                running_f1_target.extend(y.cpu().data)
                running_loss += loss.cpu().data
                #random_idx = random.randrange(0, config.batch_size)
                if te_idx % 10 == 0:
                    pred_label = torch.argmax(pred.cpu().data, dim=1)
                    real_label = y.cpu().data
                    for img_idx in range(config.batch_size):
                        if pred_label[img_idx] != real_label[img_idx]:
                            examples.append(wandb.Image(X[img_idx], caption=f'Pred: {torch.argmax(pred.cpu().data, dim=1)[img_idx]}, Real: {y.cpu().data[img_idx]}'))
            
        #batch_corss_loss /= (tr_idx+1)
        #batch_foscal_loss /= (tr_idx+1)
        batch_loss /= (tr_idx+1)
        batch_f1 = f1_score(batch_f1_target, batch_f1_pred, average='macro')
        running_loss /= (te_idx+1)
        running_acc /= (te_idx+1)
        running_f1 = f1_score(running_f1_target, running_f1_pred, average='macro')

        # result['train_loss'].append(batch_loss.cpu().data)
        # #result['train_cross_loss'].append(batch_corss_loss.cpu().data)
        # #result['train_foscal_loss'].append(batch_foscal_loss.cpu().data)
        # result['train_f1'].append(batch_f1)
        # result['valid_loss'].append(running_loss.cpu().data)
        # result['valid_acc'].append(running_acc)
        # result['valid_f1'].append(running_f1)

        scheduler.step()

        wandb.log({
            f"epoch": e,
            f"train_loss": batch_loss,
           # f"train_cross_loss": batch_corss_loss,
           # f"train_foscal_loss": batch_foscal_loss,
            f"train_f1": batch_f1,
            f"valid_loss": running_loss,
            f"valid_acc": running_acc,
            f"valid_f1": running_f1
            })
        

        if fold_index == -1:
            print(f"epoch: {e} | "
                f"train_loss:{batch_loss:.5f} | "
            #    f"train_cross_loss:{batch_corss_loss:.5f} | "
            #    f"train_foscal_loss:{batch_foscal_loss:.5f} | "
                f"train_f1:{batch_f1:.5f} |"
                f"valid_loss:{running_loss:.5f} | "
                f"valid_acc:{running_acc:.5f} | "
                f"valid_f1:{running_f1:.5f}"
                )
            wandb.log({f"epoch": e, f"Images": examples})
        else:
            print(f"fold: {fold_index} | "
                f"epoch: {e} | "
                f"train_loss:{batch_loss:.5f} | "
                f"train_f1:{batch_f1:.5f} |"
                f"valid_loss:{running_loss:.5f} | "
                f"valid_acc:{running_acc:.5f} | "
                f"valid_f1:{running_f1:.5f}"
                )
            wandb.log({f"epoch": e, f"Images_{fold_index}": examples})
        # if e % 3 == 0:
        #     print("-"*10, "Check_point", "-"*10)
        #     torch.save(model, f'../models/{model_name}_{e}_{batch_f1:.2f}.pt')
        #     print("-"*10, "Check_point Saved!!", "-"*10)

        # f1 score 기준으로 best 모델 채택
        # early_stopping_count 활용
        if running_f1 > best_metric:
            print("-"*10, "Best model changed", "-"*10)
            print("-"*10, "Model_save", "-"*10)
            if fold_index == -1:
                torch.save(model, f'../../models/{config.model_name}/{config.model_name}_{config.config_file_name}_best.pt')
            else:
                torch.save(model, f'../../models/{config.model_name}/fold_{fold_index}_{config.config_file_name}_{config.model_name}_best.pt')
            best_metric = running_f1
            best_model_dict = model.state_dict()
            print("-"*10, "Saved!!", "-"*10)
        else:
            early_stopping_count += 1

        # Loss, metric 변화 저장
        # if fold_index == -1:
        #     pd.DataFrame(result).to_csv(f'../../results/{config.model_name}/{config.model_name}_result.csv', index=False)
        # else:
        #     pd.DataFrame(result).to_csv(f'../../results/{config.model_name}/fold_{fold_index}_{config.model_name}_result.csv', index=False)

        if early_stopping_count == config.early_stopping:
            print("-"*10, "Early Stop!!!!", "-"*10)
            break
        
    return best_model_dict

if __name__ == "__main__":
    argv = sys.argv
    FILE_NAME = argv[0] # 실행시키는 파일명
    CONFIG_PATH = ""   # config file 경로
    
    try:
        # 파일명 이후 부터 입력 받는 옵션
        # help, config_path
        opts, etc_args = getopt.getopt(argv[1:], "hc:", ["help", "config_path="])
    except getopt.GetoptError:
        # 잘못된 옵션을 입력하는 경우
        print(FILE_NAME, "-c <config_path>")
        sys.exit(2)
        
    # 입력된 옵션을 적절히 변수로 입력
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(FILE_NAME, "-c <config_path>")
            sys.exit(0)
        elif opt in ("-c", "--config_path"):
            CONFIG_PATH = arg
    
    # 입력이 필수적인 옵션
    if len(CONFIG_PATH) < 1:
        print(FILE_NAME, "-c <config_path> is madatory")
        sys.exit(2)
        
    config_file_name = CONFIG_PATH.split('/')[1].split('.')[0]
    config = read_config(CONFIG_PATH)
    config.config_file_name = config_file_name
    # 나중에 config로 바꿔줄 인자들
    # For Augmentation
    aug_targets = [8, 11, 14, 17]
    transform_train = Compose([
        #transforms.CenterCrop(384),
        RandomCrop(always_apply=True, height=384, width=384, p=1.0),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
        #GaussNoise(var_limit=(1000, 1500), p=0.5), # add
        #MotionBlur(p=0.5), # add
        Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
        #T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    transform_valid = Compose([
        #transforms.CenterCrop(384),
        RandomCrop(always_apply=True, height=384, width=384, p=1.0),
        Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
        #T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])

    # 결과 및 모델 저장할 폴더
    create_dir([f'../../results/{config.model_name}', f'../../models/{config.model_name}'])
    
    config.device = 'cuda' if  torch.cuda.is_available() else 'cpu'
    print("-"*10, "Device info", "-"*10)
    print(config.device)
    print("-"*10, "-----------", "-"*10)

    # 데이터 불러오기
    print("Data Loading...")
    img_list, y_list = path_maker(config.train_csv_path, config.train_images_path, config.load_augmentation)
    
    # augmentation == True 이면 
    # 정해신 target class에 대한 이미지만 augmentation
    # [2770 2045 2490 3635 4090 3270 3324 2454 2282 4362 4908 2834 3324 2454 2292 4362 4908 2834]
    if config.augmentation:
        print("-"*10,"Start Augmentation", "-"*10)
        print("Target: ", aug_targets)
        preprocess = Preprocessing(img_list, y_list, aug_targets, config.aug_num)
        preprocess.augmentation()
        # augmentation된 이미지까지 추가된 path, label 받아오기
        img_list, y_list = path_maker(config.train_csv_path, config.train_images_path, config.load_augmentation)
        print("-"*10,"End Augmentation", "-"*10)    
    
    # unbalanced 클래스에 가중치를 주기 위한 것
    # 가장 많은 클래스 데이터 수 / 해당 클래스 데이터수
    _ , class_num = np.unique(y_list, return_counts = True)
    
    print("Class Balance: ", class_num)
    #class_num = [y_list.count(i) for i in sorted(pd.unique(y_list))]
    base_class = np.max(class_num)
    class_weigth = (base_class / np.array(class_num))
    #class_weigth = class_weigth / np.max(class_weigth)
    #print(class_num, class_weigth)
    
    # Cross validation 안할때
    if config.k_fold_num == -1:
        group_name = f'{config.model_name}'
        name = f'{config.model_name}_{config_file_name}'
        # if config.loss == 'Crossentropy_foscal':
        #         group_name = f'{config.loss}_{config.loss1_weight}_{config.loss2_weight}'
        #         name = f'{config.loss}_{config.loss1_weight}_{config.loss2_weight}'
        # else:
        #     group_name = f'{config.loss}'
        #     name = f'{config.loss}'
        wandb.init(project='Ageclassificiation', entity='kyunghyun', name=name, group=group_name, config=config, settings=wandb.Settings(start_method="fork"))
        # train, valid 데이터 분리
        # train_test_split(X, y, 훈련크기(0.8 이면 80%), stratify = (클래스 별로 분할후 데이터 추출 => 원래 데이터의 클래스 분포와 유사하게 뽑아준다) )
        # random_state는 원하는 숫자로 고정하시면 됩니다! 저는 42를 주로써서...
        train_img, valid_img, train_y, valid_y = train_test_split(img_list, y_list, train_size=0.8, 
                                shuffle=True, random_state=42, stratify=y_list)
        
        # dataset.py에서 구현한 dataset class로 훈련 데이터 정의
        train_dataset = TrainDataset_v2(train_img, train_y, transform_train)
        
        # dataset.py에서 구현한 dataset class로 평가 데이터 정의
        valid_dataset = TrainDataset_v2(valid_img, valid_y, transform_valid)
    
        # DataLoader에 넣어주기
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=3, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=3, shuffle=False)
        print(f'Train_data: {len(train_dataset)}, Valid_data:{len(valid_dataset)}')
        print("Data Loading... Success!!")
        
        print("Train Start!!")
        best_model = train(train_loader, valid_loader, class_weigth, -1, config)

        # best_model을 저장? 미구현 Train 안에서 저장 중

    # Cross validation 할때
    else:
        # K개의 corss validation 준비
        kf = StratifiedKFold(n_splits=config.k_fold_num, random_state=42, shuffle=True)
        
        print(f'{config.k_fold_num} cross validation strat...')
        
        # kf가 랜덤으로 섞어서 추출해 index들을 반환
        for fold_index, (train_idx, valid_idx) in enumerate(kf.split(img_list, y_list), 1):
            print(f'{fold_index} fold start -')
            group_name = f'{config.model_name}_fold'
            name = f'{config.model_name}_{config_file_name}_{fold_index}'
            # if config.loss == 'Crossentropy_foscal':
            #     group_name = f'{config.loss}_{config.loss1_weight}_{config.loss2_weight}_fold' 
            #     group_name = f'{config.loss}_{config.loss1_weight}_{config.loss2_weight}_fold'
            #     name = f'{config.loss}_{config.loss1_weight}_{config.loss2_weight}_{fold_index}'
            # else:
            #     group_name = f'{config.loss}_fold'
            #     name = f'{config.loss}_{fold_index}'
            run = wandb.init(project='Ageclassificiation', entity='kyunghyun', group=group_name, name=name, config=config, settings=wandb.Settings(start_method="fork"))
            # index로 array 나누기
            train_list, train_label = img_list[train_idx], y_list[train_idx]
            valid_list, valid_label = img_list[valid_idx], y_list[valid_idx]
            
            # dataset.py에서 구현한 dataset class로 훈련 데이터 정의
            train_dataset = TrainDataset_v2(train_list, train_label, transform_train)
            
            # dataset.py에서 구현한 dataset class로 평가 데이터 정의
            valid_dataset = TrainDataset_v2(valid_list, valid_label, transform_valid)
            
            # DataLoader에 넣어주기
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=3, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=3, shuffle=False)
            
            print("Train Start!!")
            best_model = train(train_loader, valid_loader, class_weigth, fold_index, config)
            run.finish()

        # best_model을 저장? 미구현 Train 안에서 저장 중