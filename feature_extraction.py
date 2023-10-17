import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()
        self.model    = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 13)
        self.model.eval()
        
        self.decoder = decoder
        self.decoder = decoder

    def forward(self, image):
        x = self.model(image)
        return x
    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x


if __name__ == '__main__':
    model = FeatureExtractor()

    # Load the saved state_dict
    saved_state_dict = torch.load('model_evaluation/model_20230813_090225/weights/model_epoch20_loss0.0818.pth', map_location=torch.device('cpu'))

    # Map the keys from the saved state_dict to the modified model
    model_state_dict = model.state_dict()
    for name, param in saved_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)

    # Save the modified model's state_dict
    torch.save(model_state_dict, 'final_model/image_model.pth')