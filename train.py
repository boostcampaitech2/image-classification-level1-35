from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from torch.utils.data import sampler

# dataset.py 안에 있는거 전부 임포트
from dataset import *

# pip install -U scikit-learn 으로 설치 가능!
# train_test_split => train, valid 셋 분할
# StratifiedKfold => Cross validation
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

if __name__ == "__main__":
    # 나중에 config로 바꿔줄 것들
    k_fold_num = -1
    load_augmentation = False
    
    train_csv_path = "../input/data/train/train.csv"
    train_images_path = "../input/data/train/images/"
    transform = transforms.Compose([
        #transforms.CenterCrop(384),
        Resize((384, 384), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    
    # device 설정
    device = 'cuda' if  torch.cuda.is_available() else 'cpu'
    print("-"*10, "Device info", "-"*10)
    print(device)
    print("-"*10, "-----------", "-"*10)

    # 데이터 불러오기
    print("Data Loading...")
    img_list, y_list = PathMaker(train_csv_path, train_images_path, load_augmentation)

    # Cross validation 안할때
    if k_fold_num == -1:
        # train, valid 데이터 분리
        # train_test_split(X, y, 훈련크기(0.8 이면 80%), stratify = (클래스 별로 분할후 데이터 추출 => 원래 데이터의 클래스 분포와 유사하게 뽑아준다) )
        # random_state는 원하는 숫자로 고정하시면 됩니다! 저는 42를 주로써서...
        train_img, valid_img, train_y, valid_y = train_test_split(img_list, y_list, train_size=0.8, 
                                shuffle=True, random_state=42, stratify=y_list)
        
        # dataset.py에서 구현한 dataset class로 훈련 데이터 정의
        train_dataset = TrainDataset_v2(train_img, train_y, transform)
        
        # dataset.py에서 구현한 dataset class로 평가 데이터 정의
        valid_dataset = TrainDataset_v2(valid_img, valid_y, transform)
    
        # DataLoader에 넣어주기
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        print(f'Train_data: {len(train_dataset)}, Valid_data:{len(valid_dataset)}')
        print("Data Loading... Success!!")
        
        # 구현 예정인 친구들...
        print("Train Start!!")
        result, best_model = train(train_loader, valid_loader)
    # Cross validation 할때
    else:
        # K개의 corss validation 준비
        kf = StratifiedKFold(n_splits=k_fold_num, random_state=42, shuffle=True)
        
        print(f'{k_fold_num} cross validation strat...')
        
        # kf가 랜덤으로 섞어서 추출해 index들을 반환
        for fold_index, (train_idx, valid_idx) in enumerate(kf.split(img_list, y_list), 1):
            print(f'{fold_index} fold start -')
                
            # index로 array 나누기
            train_list, train_label = img_list[train_idx], y_list[train_idx]
            valid_list, valid_label = img_list[valid_idx], y_list[valid_idx]
            
            # dataset.py에서 구현한 dataset class로 훈련 데이터 정의
            train_dataset = TrainDataset_v2(train_list, train_label, transform)
            
            # dataset.py에서 구현한 dataset class로 평가 데이터 정의
            valid_dataset = TrainDataset_v2(valid_list, valid_label, transform)
            
            # DataLoader에 넣어주기
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
            
            # 구현 예정인 친구들...
            print("Train Start!!")
            result, best_model = train(train_loader, valid_loader)
            
            
        
