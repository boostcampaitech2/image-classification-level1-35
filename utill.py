import os
import wandb
import configparser
import wandb
import numpy as np

# 결과 dict 및 모델 저장할 폴더 생성
def create_dir(pathes):
    for path in pathes:
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError:
           print('Error create_dir.' + path)

# Config 파일 parsing
def read_config(paths):
    config = wandb.config

    values = configparser.ConfigParser()
    values.read(paths, encoding='utf-8')

    # For Aug
    config.augmentation = values['augmentation'].getboolean('augmentation', "b")
    config.load_augmentation = values['augmentation'].getboolean('load_augmentation', "b")
    config.aug_num = int(values['augmentation']['aug_num'])
    
    # For Path
    config.train_csv_path = values['path']['train_csv_path']
    config.train_images_path = values['path']['train_images_path']
    config.model_save_path = values['path']['model_save_path']
    config.result_save_path = values['path']['result_save_path']
    config.save_name = values['path']['save_name']

    # For wandb
    config.wandb_group_name = values['wandb']['wandb_group_name']
    config.wandb_name = values['wandb']['wandb_name']
    config.wandb_entity = values['wandb']['wandb_entity']
    config.wandb_project_name = values['wandb']['wandb_project_name']

    # For 학습
    config.optimizer = values['training']['optimizer']
    config.scheduler = values['training']['scheduler']
    config.loss = values['training']['loss']
    config.loss1_weight = float(values['training']['loss1_weight'])
    config.loss2_weight = float(values['training']['loss2_weight'])
    config.model_name = values['training']['model_name']
    config.early_stopping = int(values['training']['early_stopping'])
    config.k_fold_num = int(values['training']['k_fold_num'])
    config.epoches = int(values['training']['epoches'])
    config.lr = float(values['training']['lr'])
    config.batch_size = int(values['training']['batch_size'])
    config.prediction_type =  values['training']['prediction_type']
    config.learning_type = values['training']['learning_type']
    config.num_classes = int(values['training']['num_classes'])
    
    return config

# dict에 학습 결과 기록
def logging_with_dict(result, e, batch_loss, batch_f1, running_loss, running_acc, running_f1):
    result['epoch'].append(e)
    result['train_loss'].append(batch_loss.cpu().data)
    result['train_f1'].append(batch_f1)
    result['valid_loss'].append(running_loss.cpu().data)
    result['valid_acc'].append(running_acc)
    result['valid_f1'].append(running_f1)

    return result

# wandb에 학습 결과 기록
def logging_with_wandb(e, batch_loss, batch_f1, running_loss, running_acc, running_f1, examples, fold_index):
    wandb.log({
                f"epoch": e,
                f"train_loss": batch_loss,
                f"train_f1": batch_f1,
                f"valid_loss": running_loss,
                f"valid_acc": running_acc,
                f"valid_f1": running_f1
                })
    if fold_index == -1:
        wandb.log({f"epoch": e, f"Images": examples})
    else:
        wandb.log({f"epoch": e, f"Images_{fold_index}": examples})

# console에 출력
def logging_with_sysprint(e, batch_loss, batch_f1, running_loss, running_acc, running_f1, fold_index):
    if fold_index == -1:
        print(f"epoch: {e} | "
            f"train_loss:{batch_loss:.5f} | "
            f"train_f1:{batch_f1:.5f} |"
            f"valid_loss:{running_loss:.5f} | "
            f"valid_acc:{running_acc:.5f} | "
            f"valid_f1:{running_f1:.5f}"
            )
    else:
        print(f"fold: {fold_index} | "
            f"epoch: {e} | "
            f"train_loss:{batch_loss:.5f} | "
            f"train_f1:{batch_f1:.5f} |"
            f"valid_loss:{running_loss:.5f} | "
            f"valid_acc:{running_acc:.5f} | "
            f"valid_f1:{running_f1:.5f}"
            )

# class_weigths 계산
def get_class_weights(df):
    _ , class_num = np.unique(df['class'], return_counts = True)
    print("Class Balance: ", class_num)
    base_class = np.max(class_num)
    class_weigth = (base_class / np.array(class_num))
    return class_weigth

# 필요 없는 것!
def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]