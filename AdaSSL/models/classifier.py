import torch
from pytorch_lightning import LightningModule
class SimCLR_Classifier(LightningModule):
    def __init__(self, pretained_model, num_classes=100):
        super().__init__()
        self.backbone = pretained_model.backbone
        self.backbone.requires_grad_(False)
        self.classifier = torch.nn.Linear(512, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x).flatten(start_dim=1)
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch, batch_index):
        x, y = batch
        x, y = x.to('cuda'), y.to('cuda')
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
        return optimizer

class DINO_Classifier(LightningModule):
    def __init__(self, pretained_model, num_classes=100):
        super().__init__()
        self.backbone = pretained_model.student_backbone
        self.backbone.requires_grad_(False)
        self.classifier = torch.nn.Linear(384, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x).flatten(start_dim=1)
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch, batch_index):
        x, y = batch
        x, y = x.to('cuda'), y.to('cuda')
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
        return optimizer