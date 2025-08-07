import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)
    
    def get_embedding(self, model, image_tensor, device):
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            embedding = model(image_tensor).squeeze().cpu().numpy()
        return embedding.flatten()
