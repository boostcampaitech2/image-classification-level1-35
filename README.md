# Mask image classification task
Team: 7Features (level1-35)

# Contents
1. [Requirements](#Requirements)
2. [Project files](#Project-files)
3. [Preprocessing](#Preprocessing)
4. [Train](#Train)

# Requirements
```
torch == 1.9.0+cu102
torchvision == 0.10.0+cu102
timm == 0.42.12
albumentations == 1.0.3
sklearn == 0.24.2
cv2 == 4.5.3
PIL == 8.1.0
pandas_streaming == 0.2.175
numpy == 1.19.2
pandas == 1.1.5
facenet_pytorch
retina-face
```

# Project files
* main.py
* train.py - function for train
* preprocessing.py - make cropped image wtih facenet, retina-face
* dataset.py - class and function for data load
* model.py - models for training
* optimizer.py - optimizer for training
* Loss.py - loss for training
* utill.py - Defining functions necessary for the overall process
* config.ini - Setting the necessary parameters for the overall learning process

# Preprocessing
* Using facenet and retina-face
* Crop only the human face
```python
# Change values to suit your situation.
train_path = '/opt/ml/input/data/eval/info.csv'
img_path = '/opt/ml/input/data/eval/images' 
train_data = True
PATH = '/opt/ml/input/data/train/cropped_images' if train_data else '/opt/ml/input/data/eval/cropped_images'

# Execution
python preprocessing.py
```

# Train
1. config.ini setting
```ini
[augmentation]
; augmentation do or not
augmentation = 0
; augmentation data use or not
load_augmentation = 0
; # of augmentation
aug_num = 4
; target class of augmentation
aug_targets = [8, 11, 14, 17]

; path setting
[path]
train_csv_path = ../../input/data/train/train.csv
train_images_path = ../../input/data/train/cropped_images
model_save_path = ../../ ; Create a "models" folder in that location
result_save_path = ../../ ; Create a "results" folder in that location
save_name = ;<model name>

; wandb setting
[wandb]
wandb_group_name = ;<group name>
wandb_name = ;<run name>
wandb_entity = ;<userid>
wandb_project_name = ;<projectname>

[training]
optimizer = AdamW
scheduler = CosineAnnealingLR
loss = Crossentropy_focal_labelsmoothing
loss1_weight = 0.9
loss2_weight = 0.1
loss3_weight = 0.3
model_name = swsl_resnext50_32x4d
early_stopping = 5
k_fold_num = 5
epoches = 100
lr = 1e-4
batch_size = 32
prediction_type = Age ; Mask, Age, Gender, All
learning_type = All ; Mask, None, All (E.g. A model that judges only the image with a mask in Age)
num_classes = 3 ; # of discriminant classes according to model
```

2. main execution
```
main.py -c <config file path>
E.g. main.py -c config.ini
```

# Inference
