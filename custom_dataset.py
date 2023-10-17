import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.encoder = {"Home & Garden": 0, "Baby & Kids Stuff": 1, "DIY Tools & Materials": 2, "Music, Films, Books & Games": 3, "Phones, Mobile Phones & Telecoms": 4, "Clothes, Footwear & Accessories": 5, "Other Goods": 6, "Health & Beauty": 7, "Sports, Leisure & Travel": 8, "Appliances": 9, "Computers & Software": 10, "Office Furniture & Equipment": 11, "Video Games & Consoles": 12} # Define your encoder
        self.decoder = {v: k for k, v in self.encoder.items()}   # Create a decoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx,1]
        # print(img_name)
        image = Image.open(f"Datasets/images_fb/images_cleaned/{img_name}.jpg")
        label = self.decoder[self.data.iloc[idx, 4]]
        # print(label)
        
        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = CustomDataset(csv_file='Datasets/training_data.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)