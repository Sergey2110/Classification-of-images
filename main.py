import cv2
import gc
import glob
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.models import resnet152
from tqdm import tqdm
import warnings

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.simplefilter('ignore')

DIR_TRAIN = "train/"
DIR_TEST = "test/"
PATH_TRAIN = "train/train2.csv"
PATH_TEST = "test/test2.csv"

class ImageDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __getitem__(self, idx):
        # достаем имя изображения и ее лейбл
        image_name, label = self.data_df.iloc[idx]['ID_img'], self.data_df.iloc[idx]['class']

        # читаем картинку. read the image
        image = cv2.imread(DIR_TRAIN + f"{image_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # преобразуем, если нужно. transform it, if necessary
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()

    def __len__(self):
        return len(self.data_df)


class TestImageDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.data_df.iloc[idx]['ID_img']

        # читаем картинку
        image = cv2.imread(DIR_TEST + f"{image_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # преобразуем, если нужно
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.data_df)


class Classification:
    def __init__(self):
        print("Обучающей выборки ", len(listdir(DIR_TRAIN)))
        print("Тестовой выборки ", len(listdir(DIR_TEST)))

        gc.collect()
        # задаем преобразование изображения.
        self.train_transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomResizedCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.valid_transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomResizedCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.data_df = pd.read_csv(PATH_TRAIN)
        # self.data_df.head(3)

        # разделим датасет на трейн и валидацию, чтобы смотреть на качество
        self.train_df, self.valid_df = train_test_split(self.data_df, test_size=0.2, random_state=43)
        train_dataset = ImageDataset(self.train_df, self.train_transform)
        valid_dataset = ImageDataset(self.valid_df, self.valid_transform)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=2,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=2)

        self.valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=1,
                                                   # shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=2)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dict_acc_for_batch = {"train": {}, "test": {}}
        self.dict_loss_for_batch = {"train": {}, "test": {}}


    def crossvalid(self, res_model=None, criterion=None, optimizer=None, dataset=None, k_fold=5):
        train_score = pd.Series()
        val_score = pd.Series()

        total_size = len(dataset)
        fraction = 1 / k_fold
        seg = int(total_size * fraction)
        # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset
        # index: [trll,trlr],[vall,valr],[trrl,trrr]
        for i in range(k_fold):
            trll = 0
            trlr = i * seg
            vall = trlr
            valr = i * seg + seg
            trrl = valr
            trrr = total_size

            train_left_indices = list(range(trll, trlr))
            train_right_indices = list(range(trrl, trrr))

            train_indices = train_left_indices + train_right_indices
            val_indices = list(range(vall, valr))

            train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
            val_set = torch.utils.data.dataset.Subset(dataset, val_indices)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                                       shuffle=True, num_workers=4)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=50,
                                                     shuffle=True, num_workers=4)
            train_acc = self.train(res_model, criterion, optimizer, train_loader, val_loader, 1)
            train_score.at[i] = train_acc
            # val_acc = valid(res_model, criterion, optimizer, val_loader)
            # val_score.at[i] = val_acc

        return train_score, val_score


    def plot_history(self, train_history, val_history, title='loss'):
        plt.figure()
        plt.title('{}'.format(title))
        dd = list(map(lambda x: x.cpu().detach().numpy(), train_history))
        plt.plot(dd, label='train', zorder=1)

        # points = np.array(val_history)
        steps = list(range(0, len(train_history) + 1, int(len(train_history) / len(val_history))))[1:]

        plt.scatter(steps, val_history, marker='+', s=180, c='orange', label='val', zorder=2)
        plt.xlabel('train steps')

        plt.legend(loc='best')
        plt.grid()

        plt.show()


    def train(self, res_model, criterion, optimizer, train_dataloader, test_dataloader, NUM_EPOCH=15, show_img=False):
        train_loss_log = []
        val_loss_log = []

        train_acc_log = []
        val_acc_log = []

        for epoch in tqdm(range(NUM_EPOCH)):
            self.model.train()
            train_loss = 0.
            train_size = 0

            train_pred = 0.

            print("train")
            for imgs, labels in train_dataloader:
                optimizer.zero_grad()
                # print(labels)

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                y_pred = self.model(imgs)

                loss = criterion(y_pred, labels)
                loss.backward()

                train_loss += loss.item()
                train_size += y_pred.size(0)
                train_loss_log.append(loss.data / y_pred.size(0))
                train_pred += (y_pred.argmax(1) == labels).sum()
                optimizer.step()

            train_loss_log.append(train_loss / train_size)
            train_acc_log.append(train_pred / train_size)

            self.dict_loss_for_batch["train"].update({epoch: train_loss_log[:]})
            self.dict_acc_for_batch["train"].update({epoch: train_acc_log[:]})

            # if show_img and epoch > (epoch - 2) and train_pred / train_size < 0.9:
            #     for j in range(4):
            #         show_input(imgs[j].cpu(),
            #                    title=f"{labels[j]} {list_file[list_index_val[j + i * batch_size_v]][0]}")
            #         print(f" epoch = {epoch} acc = {(train_pred / train_size) / batch_size_v}")

            val_loss = 0.
            val_size = 0
            val_pred = 0.
            self.model.eval()

            print("test")
            with torch.no_grad():
                for imgs, labels in test_dataloader:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    print(labels)

                    pred = self.model(imgs)
                    loss = criterion(pred, labels)

                    val_loss += loss.item()
                    val_size += pred.size(0)
                    val_pred += (pred.argmax(1) == labels).sum()

            val_loss_log.append(val_loss / val_size)
            val_acc_log.append(val_pred / val_size)

            self.dict_loss_for_batch["test"].update({epoch: val_loss_log[:]})
            self.dict_acc_for_batch["test"].update({epoch: val_acc_log[:]})

            clear_output()
            self.plot_history(train_loss_log, val_loss_log, 'loss')

            print('Train loss:', (train_loss / train_size) * 100)
            print('Val loss:', (val_loss / val_size) * 100)
            print('Train acc:', (train_pred / train_size) * 100)
            print('Val acc:', (val_pred / val_size) * 100)

        return train_loss_log, train_acc_log, val_loss_log, val_acc_log


    def watch_img(self):
        # посмотрим на картинки. Не забудем указать корретный путь до папки
        sns.countplot(x="class", data=self.data_df)
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Автомобиль {" "*105} Кран', fontsize=14)

        for i, name in zip(range(4), self.data_df[self.data_df['class'] == 1].sample(4, random_state=42)['ID_img']):
              img = plt.imread(DIR_TRAIN + f"{name}")
              axs[i // 2, (i % 2)].imshow(img)
              axs[i // 2, (i % 2)].axis('off')

        for i, name in zip(range(4), self.data_df[self.data_df['class'] == 0].sample(4, random_state=42)['ID_img']):
              img = plt.imread(DIR_TRAIN + f"{name}")
              axs[i // 2, (i % 2)+2].imshow(img)
              axs[i // 2, (i % 2)+2].axis('off')

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)


    def train_model(self):
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(2048, 8)
        self.model = self.model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=0.01)
        train_loss_log, train_acc_log, val_loss_log, val_acc_log = self.train(self.model,
                                                                         criterion,
                                                                         optimizer,
                                                                         self.train_loader,
                                                                         self.valid_loader,
                                                                         5)
        return train_loss_log, train_acc_log, val_loss_log, val_acc_log


    def evaluation_model(self):
        valid_predicts = []
        self.model.eval()
        for imgs, _ in tqdm(self.valid_loader):
            imgs = imgs.to(self.device)
            pred = self.model(imgs)
            pred_numpy = pred.cpu().detach().numpy()
            for class_obj in pred_numpy:
                index, max_value = max(enumerate(class_obj), key=lambda i_v: i_v[1])
                valid_predicts.append(index)
        self.valid_df["pred"] = valid_predicts
        val_accuracy = recall_score(self.valid_df['class'].values, self.valid_df['pred'].values, average="macro")
        print(f"Validation accuracy = {val_accuracy}")

        self.test_df = pd.read_csv(PATH_TEST)
        self.test_df = self.test_df.drop(["class"], axis=1)
        self.test_dataset = TestImageDataset(self.test_df, self.valid_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=32,
                                                       # shuffle=True,
                                                       pin_memory=True,
                                                       num_workers=2)


    def create_submit(self):
        self.model.eval()
        predicts = []
        for imgs in tqdm(self.test_loader):
            imgs = imgs.to(self.device)
            pred = self.model(imgs)
            for class_obj in pred:
                index, max_value = max(enumerate(class_obj), key=lambda i_v: i_v[1])
                predicts.append(index)

        self.test_df["class"] = predicts
        self.test_df.head()
        self.test_df.to_csv("submit.csv", index=False)


if __name__ == '__main__':
    classif = Classification()
    acc_loss = classif.train_model()
    classif.evaluation_model()
    classif.create_submit()
