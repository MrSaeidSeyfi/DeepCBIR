import torch
import torchvision.models as models


class ImageEmbeddingModel:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
    
    def _load_model(self):
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model.to(self.device)
    
    def get_device(self):
        return self.device


