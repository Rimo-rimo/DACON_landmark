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


#하이퍼 파라미터 튜닝
CFG = {
    'IMG_SIZE':128, #이미지 사이즈
    'EPOCHS':50, #에포크
    'LEARNING_RATE':0.0001, #학습률
    'BATCH_SIZE':64, #배치사이즈
    'SEED':41, #시드
}

wandb.init(project="DACON_landmark", entity="rimmo")
wandb.run.name = 'test'
wandb.config.update(CFG)

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
    
# transform 
# train_transform = transforms.Compose([
#                     transforms.ToPILImage(), #Numpy배열에서 PIL이미지로
#                     transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]), #이미지 사이즈 변형
#                     transforms.ToTensor(), #이미지 데이터를 tensor
#                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) #이미지 정규화              
#                     ])
# test_transform = transforms.Compose([
#                     transforms.ToPILImage(),
#                     transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#                     ])

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
                            A.CenterCrop(always_apply=False, p=1.0, height=421, width=735),
                            A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(-0.3, 0.3), interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None),
                                ],p=0.3),
                        A.OneOf([
                            A.Equalize(always_apply=False, p=1.0, mode='cv', by_channels=True),
                            A.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-11, 11), sat_shift_limit=(-13, 13), val_shift_limit=(-15, 15)),
                            A.RandomBrightness(always_apply=False, p=1.0, limit=(-0.2, 0.2)),
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
    label = data_csv["label"].tolist()
    
    return img_path, label

data_dir = '/content/drive/MyDrive/DACON_landmark/dataset/train'
train_csv = '/content/drive/MyDrive/DACON_landmark/dataset/train_4.csv'
valid_csv = '/content/drive/MyDrive/DACON_landmark/dataset/valid_4.csv'
train_img_path, train_label = get_data(data_dir, train_csv)
valid_img_path, valid_label = get_data(data_dir, valid_csv)


# Get Dataloader
#CustomDataset class를 통하여 train dataset생성
train_dataset = CustomDataset(train_img_path, train_label, train_mode=True, transforms=train_transform) 
#만든 train dataset를 DataLoader에 넣어 batch 만들기
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=2)

#vaildation 에서도 적용
vali_dataset = CustomDataset(valid_img_path, valid_label, train_mode=True, transforms=test_transform)
vali_loader = DataLoader(vali_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=2)


# 모델
class CNNclassification(torch.nn.Module):
    def __init__(self):
        super(CNNclassification, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1), #cnn layer
            nn.ReLU(), #activation function
            nn.MaxPool2d(kernel_size=2, stride=2)) #pooling layer
        
        self.fc_layer = nn.Sequential( 
            nn.Linear(3136, 10) #fully connected layer(ouput layer)
        )    
        
    def forward(self, x):
        
        x = self.layer1(x) #1층
        
        x = self.layer2(x) #2층
         
        x = self.layer3(x) #3층
        
        x = self.layer4(x) #4층
        
        x = torch.flatten(x, start_dim=1) # N차원 배열 -> 1차원 배열
        
        out = self.fc_layer(x)
        return out


# 학습 하이퍼 파라미터
model = CNNclassification().to(device)
wandb.watch(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = None

# 학습
def train(model, optimizer, train_loader, scheduler, device): 
    model.to(device)
    n = len(train_loader)
    best_acc = 0
    
    for epoch in range(1,CFG["EPOCHS"]+1): #에포크 설정
        model.train() #모델 학습
        running_loss = 0.0
            
        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), label.to(device) #배치 데이터
            optimizer.zero_grad() #배치마다 optimizer 초기화
        
            # Data -> Model -> Output
            logit = model(img) #예측값 산출
            loss = criterion(logit, label) #손실함수 계산
            
            # 역전파
            loss.backward() #손실함수 기준 역전파 
            optimizer.step() #가중치 최적화
            running_loss += loss.item()
              
        print('[%d] Train loss: %.10f' %(epoch, running_loss / len(train_loader)))
        
        if scheduler is not None:
            scheduler.step()
            
        #Validation set 평가
        model.eval() #evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
        vali_loss = 0.0
        correct = 0
        with torch.no_grad(): #파라미터 업데이트 안하기 때문에 no_grad 사용
            for img, label in tqdm(iter(vali_loader)):
                img, label = img.to(device), label.to(device)

                logit = model(img)
                vali_loss += criterion(logit, label)
                pred = logit.argmax(dim=1, keepdim=True)  #11개의 class중 가장 값이 높은 것을 예측 label로 추출
                correct += pred.eq(label.view_as(pred)).sum().item() #예측값과 실제값이 맞으면 1 아니면 0으로 합산
        vali_acc = 100 * correct / len(vali_loader.dataset)
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(vali_loss / len(vali_loader), correct, len(vali_loader.dataset), 100 * correct / len(vali_loader.dataset)))
        wandb.log({
            "Valid_Accuracy": 100 * correct / len(vali_loader.dataset),
            "Valid_Loss": vali_loss / len(vali_loader),
            })
        #베스트 모델 저장
        if best_acc < vali_acc:
            best_acc = vali_acc
            torch.save(model.state_dict(), '/content/drive/MyDrive/DACON_landmark/trained_weight/test_model.pth') #이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')


if __name__ == "__main__":
    # 학습
    train(model, optimizer, train_loader, scheduler, device)