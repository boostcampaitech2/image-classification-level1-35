import torch 
import torch.nn as nn 
import pandas as pd 
import numpy as np 
import os
import cv2
from facenet_pytorch import MTCNN


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all=True, device=device)

workers = 0 if os.name == 'nt' else 4
print(workers)


def new_train_dataset(train_path, img_path):

    raw = pd.read_csv(train_path)

    #사람별 폴더의 파일 7개 경로 가져오기
    path = []
    route = []
    
    for v in raw['path']:
        path.append(os.listdir(img_path + '/' + v))


    #새로운 데이터셋을 위한 mask, path(사진별 파일)의 feature생성하기
    mask = []
    pic_path = []

    for i_d in path:
        for pic in i_d:
            if 'aug' in pic:
                continue
            if pic[0] == 'm':
                mask.append('wear')
                pic_path.append(pic)
            elif pic[0] == 'i':
                mask.append('incorrect')
                pic_path.append(pic)
            elif pic[0] == 'n':
                mask.append('not wear')
                pic_path.append(pic)
            else:
                pass


    #기존의 데이터셋에 있는 id, age, gender, path(사람별 폴더) 개수 맞추기
    person_path = []
    for i in raw['path']:
        for j in range(7):
            person_path.append(i)

    id = []
    for i in raw['id']:
        for j in range(7):
            id.append(i)


    gender = []
    for i in raw['gender']:
        for j in range(7):
            gender.append(i)

    age = []
    for i in raw['age']:
        for j in range(7):
            age.append(i)

    #사람별 + 사진별 경로 합쳐서 최종 path 생성하기
    final_path = []
    for i in range(18900):
        final_path.append(os.path.join(img_path ,person_path[i] + '/' + pic_path[i])) 

    #새로운 데이터셋 생성하기
    df = pd.DataFrame({'id': id, 
                    'gender': gender, 
                    'age': age, 
                    'mask': mask, 
                    'path': final_path})

    return df

def get_label(df, model_type):
    if model_type == 'Mask':
        df.loc[(df['mask']=='wear'), 'class'] = 0
        df.loc[(df['mask']=='not wear'), 'class'] = 1
        df.loc[(df['mask']=='incorrect'), 'class'] = 2
    elif model_type == 'Age':
        df.loc[(df['age'] < 30), 'class'] = 0
        df.loc[(df['age'] >= 30)&(df['age']< 60), 'class'] = 1
        df.loc[(df['age'] >= 60), 'class'] = 2            
    elif model_type == 'Gender':
        df.loc[(df['gender']=='male'), 'class'] = 0
        df.loc[(df['gender']=='female'), 'class'] = 1
    # 18 클래스 생성(라벨링)
    elif model_type == 'All':
        # mask=wear
        df.loc[(df['gender']=='male')&(df['age'] < 30)&(df['mask']=='wear'), 'class'] = 0
        df.loc[(df['gender']=='male')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='wear'), 'class'] = 1
        df.loc[(df['gender']=='male')&(df['age'] >= 60)&(df['mask']=='wear'), 'class'] = 2            
        df.loc[(df['gender']=='female')&(df['age'] < 30)&(df['mask']=='wear'), 'class'] = 3
        df.loc[(df['gender']=='female')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='wear'), 'class'] = 4
        df.loc[(df['gender']=='female')&(df['age'] >= 60)&(df['mask']=='wear'), 'class'] = 5

        #mask=incorrect
        df.loc[(df['gender']=='male')&(df['age'] < 30)&(df['mask']=='incorrect'), 'class'] = 6
        df.loc[(df['gender']=='male')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='incorrect'), 'class'] = 7
        df.loc[(df['gender']=='male')&(df['age'] >= 60)&(df['mask']=='incorrect'), 'class'] = 8
        df.loc[(df['gender']=='female')&(df['age'] < 30)&(df['mask']=='incorrect'), 'class'] = 9
        df.loc[(df['gender']=='female')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='incorrect'), 'class'] = 10
        df.loc[(df['gender']=='female')&(df['age'] >= 60)&(df['mask']=='incorrect'), 'class'] = 11

        #mask=normal
        df.loc[(df['gender']=='male')&(df['age'] < 30)&(df['mask']=='not wear'), 'class'] = 12
        df.loc[(df['gender']=='male')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='not wear'), 'class'] = 13
        df.loc[(df['gender']=='male')&(df['age'] >= 60)&(df['mask']=='not wear'), 'class'] = 14            
        df.loc[(df['gender']=='female')&(df['age'] < 30)&(df['mask']=='not wear'), 'class'] = 15
        df.loc[(df['gender']=='female')&(df['age'] >= 30)&(df['age']< 60)&(df['mask']=='not wear'), 'class'] = 16
        df.loc[(df['gender']=='female')&(df['age'] >= 60)&(df['mask']=='not wear'), 'class'] = 17
    else:
        print('Wrong Prediction Type!!')
        exit(1)
    df = df.astype({'class':int})
    return df



train_path = '/opt/ml/input/data/train/train.csv'
img_path = '/opt/ml/input/data/train/images' 
model_type = 'All'

df = new_train_dataset(train_path, img_path)
df = get_label(df, model_type)


# sample input shape
image_info = cv2.imread(df['path'][0])
print('sample_image_info: ', image_info.shape)

# sample output shape
img=cv2.imread(df['path'][0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

print("box:", boxes) 
print("probability", probs)
print("landmarks", landmarks)


# read training dataset
TRAIN_PATH = "/opt/ml/input/data/train"
df_label = pd.read_csv(os.path.join(TRAIN_PATH, "train.csv"))
df_label.head()



from tqdm.notebook import tqdm
from retinaface import RetinaFace
import glob

# make empty dataframe
df_image = pd.DataFrame({})
cnt = 0 # intialize iteration count

# padding value before cropping
X_PADDING = 20 
Y_PADDING = 30 # gave more padding due to include mask on the chin & hair style

# iterrate rows for the given training dataset
# for index, row in tqdm(df_label.iterrows()):  
  
# #   # get default values
# #   train_image_paths = []
# #   id = row["id"]
# #   gender = row["gender"]
# #   race = row["race"]
# #   age = row["age"]
# #   profile = row["path"]

# #   profile_path = os.path.join(TRAIN_IMGS_DATASET_PATH, profile)
# #   # print(profile_path)

# #   # get list of images from the given profile path
# #   jpg_file_list = glob.glob(f"{profile_path}/*.jpg")
# #   jpeg_file_list = glob.glob(f"{profile_path}/*.jpeg")
# #   png_file_list = glob.glob(f"{profile_path}/*.png")
# #   list_images = jpg_file_list + jpeg_file_list + png_file_list
#   # print(list_images)

for image_file_path in df['path']:
    cnt += 1
    
    # read image and extract information using mtcnn
    img = cv2.imread(image_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
  
    # default detection with mtcnn 
    if probs[0]:
      # print("solved with mtcnn")
      # save face bounding box information
      xmin = int(boxes[0, 0]) - X_PADDING
      ymin = int(boxes[0, 1]) - Y_PADDING
      xmax = int(boxes[0, 2]) + X_PADDING
      ymax = int(boxes[0, 3]) + Y_PADDING
      
      # save landmark information
      left_eye = landmarks[0, 0]
      right_eye = landmarks[0, 1]
      mouth_left = landmarks[0, 2]
      mouth_right = landmarks[0, 3]
      nose = landmarks[0, 4]
    
    # if mtcnn fails, use retinaface
    else:
      result_detected = RetinaFace.detect_faces(image_file_path)
      # print(result_detected)

      # try retinaface to resolve,:
      if type(result_detected) == dict:
        print("resolving with retinaface: ", image_file_path)
        # save face bounding box information
        xmin = int(result_detected["face_1"]["facial_area"][0]) - X_PADDING
        ymin = int(result_detected["face_1"]["facial_area"][1]) - Y_PADDING
        xmax = int(result_detected["face_1"]["facial_area"][2]) + X_PADDING
        ymax = int(result_detected["face_1"]["facial_area"][3]) + Y_PADDING
        
        # save landmark information
        face_landmarks = result_detected["face_1"]["landmarks"]
        left_eye = face_landmarks["left_eye"]
        right_eye = face_landmarks["right_eye"]
        mouth_left = face_landmarks["mouth_left"]
        mouth_right = face_landmarks["mouth_right"]
        nose = face_landmarks["nose"]
      
      # if both of the detection fails, center crop
      elif type(result_detected) == tuple:
        print("this one is causing trouble: ", image_file_path)
        
        # manually set coordinates
        # xmin = 50
        # ymin = 100
        # xmax = 350
        # ymax = 400

        xmin = 80
        ymin = 50
        xmax = 80 + 220
        ymax = 50 + 320


        # leave landmark information empty
        face_landmarks = left_eye = right_eye = np.nan
        mouth_left = mouth_right = nose = np.nan
      
    # add row to the df_images with the extracted information
    df_image = df_image.append(
      {
        "id":df['id'],
        "gender":df['gender'],        
        "age":df['age'],
        "mask": df['mask'],
        "image_file_path": image_file_path,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "mouth_left": mouth_left,
        "mouth_right": mouth_right,
        "nose": nose
      }, ignore_index=True)
  
    # print data information every 100 iterations
    if cnt % 100 == 0:
      print(df_image.shape)
      print(df_image.info())
      print(df_image.tail())
  
df_image.head(10)
