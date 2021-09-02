
import torch
from albumentations import *

from model import *
from dataset import *
from utill import * 
from Loss import *
from optimizer import *

from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import wandb
import torch.cuda
from torch.utils.data import DataLoader
from albumentations import *
from albumentations.pytorch import ToTensorV2

def train(train_loader, valid_loader, class_weight, fold_index, config):
    # 모델 생성
    print("Model Generation...")
    model = get_model(config)
    wandb.watch(model)
    # 모델 정보 출력
    print(model)
    #summary(model, input_size=(3, 512, 384), device=device)

    # 학습
    loss_func = get_loss(config, class_weight)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    best_metric = 0
    best_model = None
    early_stopping_count = 0
    result = {
        'epoch':[],
        'train_loss':[],
        'train_f1':[],
        'valid_loss':[],
        'valid_acc':[],
        'valid_f1':[]
        }
    print("-"*10, "Training", "-"*10)

    for e in range(1, config.epoches + 1):
        pesudo_image_path = unlabeled_dataset()
    
        if e == 1:
            batch_loss, batch_f1 = train_per_epoch(train_loader, model, loss_func, optimizer, config)
        else:
            batch_loss, batch_f1 = train_per_epoch_with_pseudo_labeling(e, train_loader, model, best_model, loss_func, optimizer, config, pesudo_image_path)
        running_loss, running_acc, running_f1, examples = vlidation_per_epoch(valid_loader, model, loss_func, config)
        
        # dic로 출력
        #result = logging_with_dict(result, e, batch_loss, batch_f1, running_loss, running_acc, running_f1)
        
        # wandb log 출력
        logging_with_wandb(e, batch_loss, batch_f1, running_loss, running_acc, running_f1, examples, fold_index)
        
        # 콘솔 출력
        logging_with_sysprint(e, batch_loss, batch_f1, running_loss, running_acc, running_f1, fold_index)
        
        # lr 스케쥴러 실행
        scheduler.step()

        # f1 score 기준으로 best 모델 채택
        # early_stopping_count 활용
        if running_f1 > best_metric:
            print("-"*10, "Best model changed", "-"*10)
            print("-"*10, "Model_save", "-"*10)
            if fold_index == -1:
                torch.save(model, f'{config.model_save_path}models/{config.save_name}/{config.save_name}_best.pt')
            else:
                torch.save(model, f'{config.model_save_path}models/{config.save_name}/fold_{fold_index}_{config.save_name}_best.pt')
            best_metric = running_f1
            best_model = model
            print("-"*10, "Saved!!", "-"*10)
        else:
            early_stopping_count += 1

        # result dict 저장
        if fold_index == -1:
            pd.DataFrame(result).to_csv(f'{config.result_save_path}results/{config.save_name}/{config.save_name}_result.csv', index=False)
        else:
            pd.DataFrame(result).to_csv(f'{config.result_save_path}results/{config.save_name}/fold_{fold_index}_{config.save_name}_result.csv', index=False)

        if early_stopping_count == config.early_stopping:
            print("-"*10, "Early Stop!!!!", "-"*10)
            break
        

# 1 epoch에 대한 훈련 코드
def train_per_epoch(train_loader, model, loss_func, optimizer, config):
    scaler = GradScaler()
    model.train()
    batch_loss = 0
    batch_f1_pred = []
    batch_f1_target = []
    # train
    torch.cuda.empty_cache() 
    for tr_idx, (X, y) in enumerate(tqdm(train_loader)):
        x = X.to(config.device)
        y = y.to(config.device)
        optimizer.zero_grad()
        with autocast():
            pred = model(x)
            loss = loss_func(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_loss += loss.cpu().data
        if config.mode == 'Regression':
            batch_f1_pred.extend(pred[0].cpu().data)
            batch_f1_target.extend(y.cpu().data)
        else:
            batch_f1_pred.extend(torch.argmax(pred.cpu().data, dim=1))
            batch_f1_target.extend(y.cpu().data)

    batch_loss /= (tr_idx+1)
    batch_f1 = f1_score(batch_f1_target, batch_f1_pred, average='macro')
    return batch_loss, batch_f1

# 1 epoch에 대한 훈련 코드
def train_per_epoch_with_pseudo_labeling(epoch, train_loader, model, best_model,loss_func, optimizer, config, pesudo_image_path):
    transform_train = Compose([
        Resize(321, 258, always_apply=True, p=1.0),
        HorizontalFlip(p=0.5),
        GaussianBlur(blur_limit = (3,7), sigma_limit=0, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
        Cutout(num_holes=np.random.randint(30,50,1)[0], max_h_size=10, max_w_size=10 ,p=0.5),
        Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])
    transform_valid = Compose([
#        CenterCrop(always_apply=True, height=384, width=384, p=1.0),
        Resize(321, 258, always_apply=True, p=1.0),
        Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])


    scaler = GradScaler()
    model.train()
    batch_loss = 0
    batch_f1_pred = []
    batch_f1_target = []
    # train
    torch.cuda.empty_cache() 
    tr_idx = 0
    for (X, y) in tqdm(train_loader):
        x = X.to(config.device)
        y = y.to(config.device)
        optimizer.zero_grad()
        with autocast():
            pred = model(x)
            basic_loss = loss_func(pred, y)
            scaler.scale(basic_loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
        batch_loss += basic_loss.cpu().data

        if config.mode == 'Regression':
            batch_f1_pred.extend(pred[0].cpu().data)
            batch_f1_target.extend(y.cpu().data)
        else:
            batch_f1_pred.extend(torch.argmax(pred.cpu().data, dim=1))
            batch_f1_target.extend(y.cpu().data)

        tr_idx += 1
    
    pesudo_dataset = TestDataset(pesudo_image_path, transform=transform_valid)
    pesudo_dataloader = DataLoader(pesudo_dataset, batch_size=32, num_workers=3, shuffle=False)
    pesudo_labels = []
    with torch.no_grad():
        for pesudo_X in tqdm(pesudo_dataloader):
            pesudo_x = pesudo_X.to(config.device)
            pred = best_model.forward(pesudo_x)
            pred = pred.argmax(dim=-1)
            pesudo_labels.extend(pred.cpu().numpy())

    pesudo_dataset = TrainDataset(pesudo_image_path,  pesudo_labels, transform_train, config)
    pesudo_dataloader = DataLoader(pesudo_dataset, batch_size=32, num_workers=3, shuffle=False)


    for pesudo_X, y in tqdm(pesudo_dataloader):
        pesudo_x = pesudo_X.to(config.device)
        y = y.to(config.device)
        pseudo_pred = model(pesudo_x)
        pseudo_loss = loss_func(pseudo_pred, y)
        loss = pseudo_loss * 0.2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    batch_loss /= (tr_idx+1)
    batch_f1 = f1_score(batch_f1_target, batch_f1_pred, average='macro')
    return batch_loss, batch_f1

# 1 epoch에 대한 평가 코드
def vlidation_per_epoch(valid_loader, model, loss_func, config):
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
                loss = loss_func(pred, y)
                if config.mode == 'Regression':
                    running_acc += accuracy_score(pred[0].cpu().data, y.cpu().data)
                    running_f1_pred.extend(torch.argmax(pred.cpu().data, dim=1))
                    running_f1_target.extend(y.cpu().data)
                else:
                    running_acc += accuracy_score(torch.argmax(pred.cpu().data, dim=1), y.cpu().data)
                    running_f1_pred.extend(torch.argmax(pred.cpu().data, dim=1))
                    running_f1_target.extend(y.cpu().data)
                running_loss += loss.cpu().data
                if config.mode != 'Regression':
                    if te_idx % 10 == 0:
                        pred_label = torch.argmax(pred.cpu().data, dim=1)
                        real_label = y.cpu().data
                        for img_idx in range(len(real_label)):
                            if pred_label[img_idx] != real_label[img_idx]:
                                examples.append(wandb.Image(X[img_idx], caption=f'Pred: {torch.argmax(pred.cpu().data, dim=1)[img_idx]}, Real: {y.cpu().data[img_idx]}'))
            
        running_loss /= (te_idx+1)
        running_acc /= (te_idx+1)
        running_f1 = f1_score(running_f1_target, running_f1_pred, average='macro')

        return running_loss, running_acc, running_f1, examples