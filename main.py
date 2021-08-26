from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import func
import dataset
import model


train_path = '/opt/ml/input/data/train/train.csv'
img_path = '/opt/ml/input/data/train/images'

#mask와 picture path가 포함된 새로운 데이터셋 
df = new_train_dataset(train_path, img_path)

#labeling
df = get_label(df)

#Dataset 생성하기
mask_dataset = MaskDataset(df, transform = transforms.Compose([
        transforms.Resize((350, 350), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        transforms.RandomResizedCrop(224),
        transforms.Grayscale(num_output_channels=3) > 감마조절
        ]))

#train/val data split
batch_size  = 16 #mixed precision(동일한 배치사이즈>메모리 적게차지)(wandb 및 ray)
random_seed = 4
random.seed(random_seed)
torch.manual_seed(random_seed)


train_idx, val_idx = train_test_split(mask_dataset[0], mask_dataset[1], test_size=0.2, random_state=random_seed)
datasets = {}
datasets['train'] = Subset(mask_dataset, train_idx)
datasets['valid'] = Subset(mask_dataset, val_idx)


#dataloader 생성하기
dataloaders, batch_num = {}, {}
dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'],
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=4)
dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'],
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=4)
batch_num['train'], batch_num['valid'] = len(dataloaders['train']), len(dataloaders['valid'])







device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = model.to(device)


# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss()


optimizer_ft = optim.SGD(model.parameters(), lr = 0.005, momentum=0.9, weight_decay=1e-4)
# optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.05)
# optimizer_ft =torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

lmbda = lambda epoch: 0.98739
exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)
# exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=50, eta_min=0)

model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)


def train_model(pretrained_model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    # best_model_wts = model.state_dict().clone().detach().requires_grad_(True)
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    
    n_iter=0
    epoch_f1 = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 에폭마다 train/valid
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train() 
            else:
                model.eval() 

            running_loss, running_corrects, num_cnt = 0.0, 0, 0
            
            # Iterate
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # parameter 초기화
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward, weight update
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

   
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))
            
            epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            n_iter += 1
	

            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.1f'%(best_idx, best_acc))

        epoch_f1 = epoch_f1/n_iter
        print(f"{epoch_f1:.4f}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), 'mask_model.pt') #모델 파라미터 저장
    torch.save(model, 'mask_model.pt') #모델 자체 저장
    print('model saved')
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc
