import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch

DIR_TRAIN = "train/"
DIR_TEST = "test/"


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
