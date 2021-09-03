import os
import sys, getopt

from dataset import *
from Loss import *
from train import *
from utill import *

from albumentations import *
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

def single_inference(MODEL_PATH, SAVE_PATH, loader):
    device = torch.device('cuda')
    if SAVE_PATH == "":
        save_name = './results.csv'
    else:
        save_name = SAVE_PATH

    model = torch.load(MODEL_PATH)
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_prediction = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_prediction.extend(pred.cpu().numpy())

    submission['ans'] = all_prediction
    submission.to_csv(save_name, index=False)
    print('test inference is done!')

def folds_inference(MODEL_PATH, SAVE_PATH, loader):
    device = torch.device('cuda')
    if SAVE_PATH == "":
        save_name = './results.csv'
    else:
        save_name = SAVE_PATH

    all_predictions = []
    for model_path in os.listdir(MODEL_PATH):
        print(f'Inference {model_path}')
        model = torch.load(model_path).to(device)
        model.eval()
        # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
        all_prediction = []
        for images in tqdm(loader):
            with torch.no_grad():
                images = images.to(device)
                pred = model(images)
                all_prediction.extend(pred.cpu().numpy())
        all_predictions.append(all_prediction)

    print(np.array(all_predictions).shape)
    all_predictions = np.array(all_predictions)
    master_predictions = all_predictions[0] + all_predictions[1] + all_predictions[2] + all_predictions[3] + all_predictions[4]
    ensemble_result = np.argmax(master_predictions, axis=1)
    print(ensemble_result.shape)

    submission['ans'] = ensemble_result
    submission.to_csv(save_name, index=False)
    print('test inference is done!')

if __name__ == "__main__":
    argv = sys.argv
    FILE_NAME = argv[0] # 실행시키는 파일명
    MODEL_PATH = ""     # 모델 경로
    DATA_PATH = ""      # DATA경로
    SAVE_PATH = ""      # 결과 저장할 경로
    INFERENCE_TYPE = "none" # 단일모델이면 none / 여러모델이면 fold-> 한폴더에 .pt 파일 집어 넣기

    try:
        # 파일명 이후 부터 입력 받는 옵션
        # help, config_path
        opts, etc_args = getopt.getopt(argv[1:], "hm:d:s:t:", ["help", "config_path=", "model_path=", "save_path=","inference_type="])
    except getopt.GetoptError:
        # 잘못된 옵션을 입력하는 경우
        print(FILE_NAME, "-m <model path> -d <file path> -s <save path> -t <inference_type>")
        sys.exit(2)
        
    # 입력된 옵션을 적절히 변수로 입력
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(FILE_NAME, "-m <model path> -d <file path> -s <save path> -t <inference_type>")
            sys.exit(0)
        elif opt in ("-m", "--model_path"):
            MODEL_PATH = arg
        elif opt in ("-d", "--data_path"):
            DATA_PATH = arg
        elif opt in ("-s", "--save_path"):
            SAVE_PATH = arg
        elif opt in ("-t", "--inference_type"):
            INFERENCE_TYPE = arg.tolower()
    
    # 입력이 필수적인 옵션
    if len(MODEL_PATH) < 1:
        print(FILE_NAME, "-m <model path> is madatory")
        sys.exit(2)
    if len(DATA_PATH) < 1:
        print(FILE_NAME, "-d <data path> is madatory")
        sys.exit(2)

    # 데이터 불러오기
    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(DATA_PATH)
    image_dir = os.path.join(test_dir, 'images')
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform_valid = Compose([
            CenterCrop(height=321, width=384, p=1.0),
            Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
    ])

    dataset = TestDataset(image_paths, transform_valid)
    loader = DataLoader(
        dataset,
        shuffle=False
    )

    if INFERENCE_TYPE == 'none':
        df = single_inference(MODEL_PATH, SAVE_PATH, loader)
    else:
        df = folds_inference(MODEL_PATH, SAVE_PATH, loader)

    print('-'*10,"Finished!!!",'-'*10)








