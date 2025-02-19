import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.transforms as transforms
from PIL import Image


class Predictor():
    def __init__(self, weight_path: str):
        self.model = torch.load(weight_path, map_location='cpu')
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

    @torch.no_grad()
    def inference(self, img: Image.Image) -> int:
        self.model.eval()

        img = self.transform(img)
        img = img.unsqueeze(0)

        output = self.model(img)
            
        _, pred = torch.max(output, dim=1)
        
        return int(pred)


if __name__ == "__main__":
    predictor = Predictor(weight_path='./model_weight/lenet.pt')
    print(predictor)
