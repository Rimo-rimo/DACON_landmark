import pandas as pd
import os
import random
import numpy as np
from tqdm.auto import tqdm
import cv2
import torch
import torch.nn as nn
import torchvision.datasets as datasets # 이미지 데이터셋 집합체
import torchvision.transforms as transforms # 이미지 변환 툴
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim # 최적화 알고리즘들이 포함힘
import wandb

import albumentations as A
from albumentations import *
from albumentations.pytorch import ToTensorV2

CFG = {
    'IMG_SIZE':128, #이미지 사이즈
    'EPOCHS':50, #에포크
    'LEARNING_RATE':0.0001, #학습률
    'BATCH_SIZE':64, #배치사이즈
    'SEED':41, #시드
}

# 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # Set the GPU 2 to use, 멀티 gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#GPU 체크 및 할당
if torch.cuda.is_available():    
    #device = torch.device("cuda:0")
    print('Device:', device)
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


# Seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None): #필요한 변수들을 선언
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def __getitem__(self, index): #index번째 data를 return
        img_path = self.img_path_list[index]
        # Get image data
        #print(img_path)
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image
    
    def __len__(self): #길이 return
        return len(self.img_path_list)

train_transform = A.Compose([
                        A.Resize(always_apply=False, p=1.0, height=540, width=960, interpolation=0),
                        A.GaussNoise(always_apply=False, p=0.3, var_limit=(159.3, 204.6)),
                        A.MotionBlur(always_apply=False, p=0.3, blur_limit=(8, 11)),
                        A.OneOf([
                            A.Rotate(always_apply=False, p=1.0, limit=(-14, 14), interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None),
                            A.HorizontalFlip(always_apply=False, p=1.0),
                                ],p=0.5),
                        A.OneOf([
                            A.ElasticTransform(always_apply=False, p=1.0, alpha=1.0, sigma=50.0, alpha_affine=50.0, interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None, approximate=False),
                            A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.30, 0.30), shift_limit=(-0.05, 0.05), interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None),
                            A.RandomResizedCrop(always_apply=False, p=1.0, height=540, width=960, scale=(0.5, 1.0), ratio=(0.75, 1.3), interpolation=0),
                            A.RandomSizedCrop(always_apply=False, p=1.0, min_max_height=(540, 540), height=540, width=960, w2h_ratio=1.0, interpolation=0),
                            A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(-0.3, 0.3), interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None),
                                ],p=0.3),
                        A.OneOf([
                            A.Equalize(always_apply=False, p=1.0, mode='cv', by_channels=True),
                            A.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-11, 11), sat_shift_limit=(-13, 13), val_shift_limit=(-15, 15))
                                ],p=0.3),             
                        A.CoarseDropout(always_apply=False, p=0.5, max_holes=20, max_height=8, max_width=8, min_holes=20, min_height=8, min_width=8),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                            ])
test_transform = A.Compose([
                        A.Resize(always_apply=False, p=1.0, height=540, width=960, interpolation=0),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                            ])

# 데이터 로드
def get_data(data_dir, data_csv):
    data_csv = pd.read_csv(data_csv)
    img_path = data_dir +"/"+ data_csv["file_name"]
    img_path = img_path.tolist()
    
    return img_path

data_dir = '/content/drive/MyDrive/DACON_landmark/dataset/test'
test_csv = '/content/drive/MyDrive/DACON_landmark/dataset/test.csv'

test_img_path, train_label = get_data(data_dir, test_csv)

test_dataset = CustomDataset(test_img_path, None, train_mode=False, transforms=test_transform)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=2)

def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.to(device)

            pred_logit = model(img)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
    return model_pred

# Validation Accuracy가 가장 뛰어난 모델을 불러옵니다.
checkpoint = torch.load('/content/drive/MyDrive/DACON_landmark/trained_weight/test_model_epoch33.pth')
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(device)
model.load_state_dict(checkpoint)

# Inference
preds = predict(model, test_loader, device)

submission = pd.read_csv('/content/drive/MyDrive/DACON_landmark/dataset/sample_submission.csv')
submission['label'] = preds
submission.to_csv('/content/drive/MyDrive/DACON_landmark/submission_file/test_model_epoch33_test.csv', index=False)