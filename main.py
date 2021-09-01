import os
import sys, getopt
from dataset import *
from Loss import *
from train import *
from utill import *

from albumentations import *
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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
  
    # trasform
    transform_train = Compose([
        # Resize(width = 384, height = 384),
        # RandomCrop(always_apply=True, height=384, width=384, p=1.0),
        CenterCrop(always_apply=True, height=384, width=384, p=1.0),
        Resize(width = 244, height = 244),
        HorizontalFlip(p=0.5),
        GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
        Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])
    transform_valid = Compose([
        CenterCrop(always_apply=True, height=384, width=384, p=1.0),
        Resize(width = 244, height = 244),
        Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])

    # 결과 및 모델 저장할 폴더
    create_dir([f'{config.result_save_path}results/{config.save_name}', f'{config.model_save_path}models/{config.save_name}'])

    config.device = 'cuda' if  torch.cuda.is_available() else 'cpu'
    print("-"*10, "Device info", "-"*10)
    print(config.device)
    print("-"*10, "-----------", "-"*10)

    # 데이터 불러오기
    print("Data Loading...")
    # img_list, y_list = path_maker(config.train_csv_path, config.train_images_path, config.load_augmentation)
    df = new_train_dataset(config.train_csv_path, config.train_images_path)
    df = get_label(df, config.prediction_type)

    if config.k_fold_num != -1:
        folds = make_fold(config.k_fold_num, df)
    print(df)
    #print(np.array(folds).shape)

    # augmentation == True 이면 
    # 정해신 target class에 대한 이미지만 augmentation
    # [2770 2045 2490 3635 4090 3270 3324 2454 2282 4362 4908 2834 3324 2454 2292 4362 4908 2834]
    # if config.augmentation:
    #     print("-"*10,"Start Augmentation", "-"*10)
    #     print("Target: ", config.aug_targets)
    #     preprocess = Preprocessing(img_list, y_list, config.aug_targets, config.aug_num)
    #     preprocess.augmentation()
    #     # augmentation된 이미지까지 추가된 path, label 받아오기
    #     img_list, y_list = path_maker(config.train_csv_path, config.train_images_path, config.load_augmentation)
    #     print("-"*10,"End Augmentation", "-"*10)    
    
    # unbalanced 클래스에 가중치를 주기 위한 것
    # 가장 많은 클래스 데이터 수 / 해당 클래스 데이터수
    
    # Cross validation 안할때
    if config.k_fold_num == -1:
        group_name = f'{config.model_name}'
        name = f'{config.model_name}_{config_file_name}'
        wandb.init(project=config.wandb_project_name, group=config.wandb_group_name, name=config.wandb_name, config=config, settings=wandb.Settings(start_method="fork"))
        # train, valid 데이터 분리
        # train_test_split(X, y, 훈련크기(0.8 이면 80%), stratify = (클래스 별로 분할후 데이터 추출 => 원래 데이터의 클래스 분포와 유사하게 뽑아준다) )
        # random_state는 원하는 숫자로 고정하시면 됩니다! 저는 42를 주로써서...
    
        valid_ids = df.groupby('id')['id'].sample(n=1).sample(n=540, random_state=42, replace=False)
        train_list, train_label, valid_list, valid_label = make_train_list(df, config, valid_ids)
        
        class_weigth = get_class_weights(train_label)

        # dataset.py에서 구현한 dataset class로 훈련 데이터 정의
        train_dataset = TrainDataset(np.array(train_list), np.array(train_label), transform_train)
        
        # dataset.py에서 구현한 dataset class로 평가 데이터 정의
        valid_dataset = TrainDataset(np.array(valid_list), np.array(valid_label), transform_valid)
    
        # DataLoader에 넣어주기
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=3, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=3, shuffle=False)
        print(f'Train_data: {len(train_dataset)}, Valid_data:{len(valid_dataset)}')
        print("Data Loading... Success!!")
        
        print("Train Start!!")
        best_model = train(train_loader, valid_loader, class_weigth, -1, config)

    # Cross validation 할때
    else:       
        print(f'{config.k_fold_num} cross validation strat...')
        
        # kf가 랜덤으로 섞어서 추출해 index들을 반환
        for fold_index, valid_ids in enumerate(folds):
            print(f'{fold_index} fold start -')
            group_name = f'{config.model_name}_fold'
            name = f'{config.wandb_name}_{fold_index}'

            run = wandb.init(project=config.wandb_project_name, group=config.wandb_group_name, name=name, config=config, settings=wandb.Settings(start_method="fork"))
            
            train_list, train_label, valid_list, valid_label = make_train_list(df, config, valid_ids)
            
            print(f'Train_Data: {train_list.shape}, Validation_Data: {valid_list.shape}')

            class_weigth = get_class_weights(train_label)

            # dataset.py에서 구현한 dataset class로 훈련 데이터 정의
            train_dataset = TrainDataset(np.array(train_list), np.array(train_label), transform_train)
            
            # dataset.py에서 구현한 dataset class로 평가 데이터 정의
            valid_dataset = TrainDataset(np.array(valid_list), np.array(valid_label), transform_valid)
            
            # DataLoader에 넣어주기
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=3, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=3, shuffle=False)
            
            print("Train Start!!")
            best_model = train(train_loader, valid_loader, class_weigth, fold_index, config)
            run.finish()
