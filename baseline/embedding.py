from turtle import forward
from torch import embedding, nn
from torchvision import models

class EmbeddingNet(nn.Module):
    def __init__(self, backbone=None, embedding_size=128):
        super().__init__()
        self.backbone = backbone
        self.embedding_size = embedding_size
        if self.backbone is None:
            self.backbone = models.resnet50(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.embedding_size)
    
    def forward(self, x):
        x = self.backbone(x)
        return x

class SoftmaxEmbeddingNet(EmbeddingNet):
    def __init__(self, num_classes, backbone=None, embedding_size=128):
        super().__init__(backbone, embedding_size)
        self.num_classes = num_classes
        self.final_layer = nn.Linear(self.embedding_size, self.num_classes)

    def forward(self, x):
        x = self.backbone(x)
        if self.training:
            x = self.final_layer(x)
        return x
            