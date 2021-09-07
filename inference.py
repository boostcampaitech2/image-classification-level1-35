import os
import glob
import sys, getopt

from dataset import *
from Loss import *
from train import *
from utill import *

from albumentations import *
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

def inference(model_path, save_path, loader):
    device = torch.device('cuda')
    if save_path == "":
        save_name = './results.csv'
    else:
        save_name = save_path

    all_predictions = []
    # model_path ~~~.pt / MODEL_PATH ~~~ -> ~~~.pt 1개
    for idx, model_pt_path in enumerate(os.listdir(model_path)):
        print('-'*10, f"Fold {idx} Start", '-'*10)
        print(f'Inference {model_pt_path}')
        model = torch.load(os.path.join(model_path, model_pt_path)).to(device)
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

    master_predictions = np.sum(all_predictions, axis=0)
    ensemble_result = np.argmax(master_predictions, axis=1)
    print(ensemble_result.shape)

    submission['ans'] = ensemble_result
    submission.to_csv(save_name, index=False)
    print('test inference is done!')

if __name__ == "__main__":
    argv = sys.argv
    file_name = argv[0] # 실행시키는 파일명
    config_path = ""    # config 파일 경로

    try:
        # 파일명 이후 부터 입력 받는 옵션
        # help, config_path
        opts, etc_args = getopt.getopt(argv[1:], "hc:", ["help", "config_path="])
    except getopt.GetoptError:
        # 잘못된 옵션을 입력하는 경우
        print(file_name, "-c <config path>")
        sys.exit(2)
        
    # 입력된 옵션을 적절히 변수로 입력
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(file_name, "-c <config path>")
            sys.exit(0)
        elif opt in ("-c", "--config_path"):
            config_path = arg
    
    # 입력이 필수적인 옵션
    if len(config_path) < 1:
        print(file_name, "-c <config path> is madatory")
        sys.exit(2)

    config = read_config(config_path)

    # 데이터 불러오기
    print('-'*10, "Data loading", '-'*10)
    submission = pd.read_csv(config.eval_csv_path)
    image_paths = [p for p in glob.glob(f'{config.eval_image_path}/**', recursive=True) if 'aug' not in p]
    transform_valid = Compose([
            Resize(height=config.image_height, width=config.image_width, p=1.0),
            Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
    ])

    dataset = TestDataset(image_paths, transform_valid)
    loader = DataLoader(
        dataset,
        shuffle=False
    )
    print('-'*10, "Data loading complete", '-'*10)

    print('-'*10, "Inference Start", '-'*10)
    
    inference(config.model_path, config.inference_result_save_path, loader)

    print('-'*10,"Finished!!!",'-'*10)








