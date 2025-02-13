import torch
import torchvision

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.profilers import PyTorchProfiler

class SimCLR(LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        resnet.fc = torch.nn.Identity()
        self.backbone = resnet
        self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
        self.criterion = loss.NTXentLoss()

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z

    def training_step(self, batch, batch_index):
        (view0, view1), _, _ = batch
        view0, view1 = view0.to('cuda'), view1.to('cuda')
        z0 = self.forward(view0)
        z1 = self.forward(view1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

# Prepare transform that creates multiple random views for every image.
transform = transforms.SimCLRTransform(input_size=32, cj_prob=0.5)

# Download and create the CIFAR-100 dataset.
cifar100_dataset = torchvision.datasets.CIFAR100(
    root='./datasets',
    train=True,
    download=True,
    transform=transform
)

# Create a dataset from your image folder.
dataset = LightlyDataset.from_torch_dataset(cifar100_dataset)

# Build a PyTorch dataloader.
dataloader = torch.utils.data.DataLoader(
    dataset,  # Pass the dataset to the dataloader.
    batch_size=128,  # A large batch size helps with the learning.
    shuffle=True,  # Shuffling is important!
    pin_memory=True,
    num_workers=4,
)

model = SimCLR()
profiler = PyTorchProfiler(
    dirpath="./log",
    filename="profiler_output",
    emit_nvtx=True,
    record_functions=True,
)
trainer = Trainer(max_epochs=5, devices=1, accelerator="gpu", profiler=profiler)
trainer.fit(model, dataloader)