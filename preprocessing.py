# 출처: 토론 게시판 
import pandas as pd 
import numpy as np 
import os
import cv2
from facenet_pytorch import MTCNN
from tqdm.auto import tqdm
from retinaface import RetinaFace
from dataset import *
import sys, getopt
import glob

def get_bound_box(df):
    # make empty dataframe
    df_image = pd.DataFrame({})
    cnt = 0 # intialize iteration count

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(keep_all=True, device=device)

    # padding value before cropping
    X_PADDING = 20 
    Y_PADDING = 30 # gave more padding due to include mask on the chin & hair style

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
            "image_file_path": image_file_path,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "mouth_left": mouth_left,
            "mouth_right": mouth_right,
            "nose": nose,
        }, ignore_index=True)

    return df_image

def make_cropped_image(df_image, save_path, train_data):
    for index, row in tqdm(df_image.iterrows()):
        # read stored data 
        image_file_path = row["image_file_path"]
        xmin = int(row["xmin"])
        ymin = int(row["ymin"])
        xmax = int(row["xmax"])
        ymax = int(row["ymax"])
        #profile = str(row["profile"])

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # place cropped image under profile folder
        profile = image_file_path.split("/")[-2]
        image_name =  image_file_path.split("/")[-1]
        image = cv2.imread( image_file_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # omit BGR2RGB process: not using mtcnn

        # prevent tile out of box error
        if xmin<0:xmin=0
        if ymin<0:ymin=0
        if xmax>384:xmax=384
        if ymax>512:ymax=512

        # crop and save cropped images
        cropped_img = image[ymin:ymax, xmin:xmax, :] # y axis, x axis, all 3 channels
        # print(profile_name, image_name)
        if train_data:
            cv2.imwrite(os.path.join(save_path, profile, image_name), cropped_img)
        else:
            cv2.imwrite(os.path.join(save_path, image_name), cropped_img)

if __name__ == "__main__":
    argv = sys.argv
    file_name = argv[0] # 실행시키는 파일명
    image_path = ''
    data_type = -1
    # default
    
    try:
        # 파일명 이후 부터 입력 받는 옵션
        # help, config_path
        opts, etc_args = getopt.getopt(argv[1:], "ht:i:s:", ["help", "train_eval=", "image_path=", "save_path="])
    except getopt.GetoptError:
        # 잘못된 옵션을 입력하는 경우
        print(file_name, "-t <train or eval> -i <image_folder_path> -s <save_path>")
        sys.exit(2)
        
    # 입력된 옵션을 적절히 변수로 입력
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(file_name, "-t <train or eval> -i <image_folder_path> -s <save_path>")
            sys.exit(0)
        elif opt in ("-t", "--train_eval"):
            if arg not in ['train', 'eval']:
                print("-t <train or eval> Wrong Input!!")
                sys.exit(2)
            data_type = True if arg == 'train' else False
            save_path = '/opt/ml/input/data/train/cropped_images' if data_type else '/opt/ml/input/data/eval/cropped_images'
        elif opt in ("-i", "--image_path"):
            image_path = arg
        elif opt in ("-s", "--save_path"):
            save_path = arg
    
    # 입력이 필수적인 옵션
    if data_type == -1:
        print(file_name, "-t <train or eval> is madatory")
        sys.exit(2)

    if image_path == -1:
        print(file_name, "-i <image_folder_path> is madatory")
        sys.exit(2)

    l = [p for p in glob.glob(f'{image_path}/**', recursive=True) if 'aug' not in p]
    df = pd.DataFrame()
    df['path'] = l

    print(df.head())
    df_image = get_bound_box(df)
    make_cropped_image(df_image, save_path, data_type)
    print('-'*10,"Finished!!!",'-'*10)


    

