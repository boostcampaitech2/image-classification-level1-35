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
    file_name = argv[0] # 실행시키는 파일명
    config_path = ""   # config file 경로
    
    try:
        # 파일명 이후 부터 입력 받는 옵션
        # help, config_path
        opts, etc_args = getopt.getopt(argv[1:], "hc:", ["help", "config_path="])
    except getopt.GetoptError:
        # 잘못된 옵션을 입력하는 경우
        print(file_name, "-c <config_path>")
        sys.exit(2)
        
    # 입력된 옵션을 적절히 변수로 입력
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(file_name, "-c <config_path>")
            sys.exit(0)
        elif opt in ("-c", "--config_path"):
            config_path = arg
    
    # 입력이 필수적인 옵션
    if len(config_path) < 1:
        print(file_name, "-c <config_path> is madatory")
        sys.exit(2)
        
    config_file_name = config_path.split('/')[1].split('.')[0]
    config = read_config(config_path)
    config.config_file_name = config_file_name
    config.mode = 'Classification' #'Regression'
    # trasform
    transform_train = Compose([
        Resize(height=config.image_height, width=config.image_width, always_apply=True, p=1.0),
        HorizontalFlip(p=0.5),
        GaussianBlur(blur_limit = (3,7), sigma_limit=0, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
        Cutout(num_holes=np.random.randint(30,50,1)[0], max_h_size=10, max_w_size=10 ,p=0.5),
        Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])
    transform_valid = Compose([
        Resize(height=config.image_height, width=config.image_width, always_apply=True, p=1.0),
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
    df = new_train_dataset(config.train_csv_path, config.train_images_path, config)
    df = get_label(df, config.prediction_type)

    if config.prediction_type == 'Age' and config.Age_external_data_load:
        age_df = read_age_data()

    
    folds = make_fold(config.k_fold_num, df)

    print(f'{config.k_fold_num} cross validation strat...')
        
    # kf가 랜덤으로 섞어서 추출해 index들을 반환
    for fold_index, valid_ids in enumerate(folds):
        print(f'{fold_index} fold start -')
        # 각 fold별 best model 저장할 배열
        models = []
        # wandb 설정
        group_name = f'{config.model_name}_fold'
        name = f'{config.wandb_name}_{fold_index}'
        run = wandb.init(project=config.wandb_project_name, entity=config.wandb_entity, group=config.wandb_group_name, name=name, config=config, settings=wandb.Settings(start_method="fork"))
        
        train_list, train_label, valid_list, valid_label = make_train_list(df, config, valid_ids)
        
        if config.prediction_type == 'Age' and config.Age_external_data_load:
            train_list = pd.concat([train_list, age_df['path']], axis=0)
            train_label = pd.concat([train_label, age_df['class']], axis=0)

        class_weight = get_class_weights(train_label)

        # dataset.py에서 구현한 dataset class로 훈련 데이터 정의
        train_dataset = TrainDataset(np.array(train_list), np.array(train_label), transform_train, config)
        
        # dataset.py에서 구현한 dataset class로 평가 데이터 정의
        valid_dataset = TrainDataset(np.array(valid_list), np.array(valid_label), transform_valid, config)
        
        # DataLoader에 넣어주기
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=3, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=3, shuffle=False)
        
        print(f'Train_Data: {train_list.shape}, Validation_Data: {valid_list.shape}')
        print("Train Start!!")
        model = train(train_loader, valid_loader, class_weight, fold_index, config)
        models.append(model)
        run.finish()     
    