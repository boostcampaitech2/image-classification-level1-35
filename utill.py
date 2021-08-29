import os
import wandb
import configparser

def create_dir(pathes):
    for path in pathes:
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError:
           print('Error create_dir.' + path)

def read_config(paths):
    config = wandb.config

    values = configparser.ConfigParser()
    values.read(paths, encoding='utf-8')

    config.augmentation = values['augmentation'].getboolean('augmentation', "b")
    config.load_augmentation = values['augmentation'].getboolean('load_augmentation', "b")
    config.aug_num = int(values['augmentation']['aug_num'])
    #config.aug_targets = values['agmentation']['augmentation']
    
    # For 학습
    config.optimizer = values['training']['optimizer']
    config.scheduler = values['training']['scheduler']
    config.loss = values['training']['loss']
    config.loss1_weight = float(values['training']['loss1_weight'])
    config.loss2_weight = float(values['training']['loss2_weight'])
    config.model_name = values['training']['model_name']
    config.ealry_stopping = int(values['training']['ealry_stopping'])
    config.k_fold_num = int(values['training']['k_fold_num'])
    config.epoches = int(values['training']['epoches'])
    config.lr = float(values['training']['lr'])
    config.batch_size = int(values['training']['batch_size'])
    config.train_csv_path = values['training']['train_csv_path']
    config.train_images_path = values['training']['train_images_path']

    return config

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]