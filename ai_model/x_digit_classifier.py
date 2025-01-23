import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DigitClassifierService:
    def __init__(self, model_path='digit_classifier.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DigitClassifier().to(self.device)
        
        # Load the pre-trained model if exists
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print("Warning: Model file not found. Using untrained model.")
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def preprocess_image(self, base64_string):
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
            
        # Decode base64 to image
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Transform image
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, base64_image):
        with torch.no_grad():
            tensor = self.preprocess_image(base64_image)
            output = self.model(tensor)
            prediction = output.max(1, keepdim=True)[1].item()
            probabilities = F.softmax(output, dim=1)[0]
            confidence = probabilities[prediction].item()
            
            return {
                'digit': prediction,
                'confidence': float(confidence)
            } 