import torch
import torch.profiler
import torchvision

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads

import sys
num_workers = int(sys.argv[1])
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
pin_memory = sys.argv[4].lower() == 'true'

# Create a PyTorch module for the SimCLR model.
class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512,  # Resnet18 features have 512 dimensions.
            hidden_dim=512,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z


# Use a resnet backbone from torchvision.
backbone = torchvision.models.resnet18()
# Ignore the classification head as we only want the features.
backbone.fc = torch.nn.Identity()

# Build the SimCLR model.
model = SimCLR(backbone).to('cuda')

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
#dataset = LightlyDataset(input_dir="./my/cute/cats/dataset/", transform=transform)
dataset = LightlyDataset.from_torch_dataset(cifar100_dataset)


# Build a PyTorch dataloader.
dataloader = torch.utils.data.DataLoader(
    dataset,  # Pass the dataset to the dataloader.
    batch_size=batch_size,  # A large batch size helps with the learning.
    shuffle=True,  # Shuffling is important!
    pin_memory=pin_memory,  # Use pin memory for faster data transfer to GPU.
    num_workers=num_workers,
)

# Lightly exposes building blocks such as loss functions.
criterion = loss.NTXentLoss(temperature=0.5).to('cuda')

# Get a PyTorch optimizer.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6)

import time

st = time.perf_counter_ns()
### 75.833220934
for epoch in range(epochs):
    for (view0, view1), targets, filenames in dataloader:
        with torch.profiler.record_function("data_loading"):
            #pass
            view0, view1 = view0.to('cuda'), view1.to('cuda')
            targets = targets.to('cuda')
end = time.perf_counter_ns()
initial_time = end - st
#print(end - st)
with open("time_log.txt", "a") as f:
    f.write(f"Setting - Epoch: {epochs} #Workers: {num_workers} BS: {batch_size} pin: {pin_memory} --- Loading time: {initial_time} ns\n")
#i=0
st = time.perf_counter_ns()
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
# Train the model.
    for epoch in range(epochs):
        for (view0, view1), targets, filenames in dataloader:
            with torch.profiler.record_function("data_loading"):
                view0, view1 = view0.to('cuda'), view1.to('cuda')
                targets = targets.to('cuda')
            with torch.profiler.record_function("model_inference"):
                z0 = model(view0)
                z1 = model(view1)
            with torch.profiler.record_function("loss_calculation"):
                loss = criterion(z0, z1)
            with torch.profiler.record_function("backpropagation"):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            prof.step()
            #i+=1
            #print(f"epoch: {epoch} loss: {loss.item():.5f}")
#print(i)
end = time.perf_counter_ns()
print(f"Total time: {end-st} ns")
with open("time_log.txt", "a") as f:
    f.write(f"Setting - Epoch: {epochs} #Workers: {num_workers} BS: {batch_size} pin: {pin_memory} --- Entire time: {end-st} ns\n")