import copy

import torch
import torch.profiler
import torchvision
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightly.data import LightlyDataset
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from lightly.transforms import utils
from sklearn.metrics import accuracy_score
import sys
num_workers = int(sys.argv[1])
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
pin_memory = sys.argv[4].lower() == 'true'

class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z
    
    @property
    def backbone(self):
        return self.student_backbone

class Classifier(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone

        # freeze the backbone
        deactivate_requires_grad(backbone)

        # create a linear layer for our downstream classification model
        self.fc = nn.Linear(384, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_fc", loss)
        return loss

    def on_train_epoch_end(self):
        self.custom_histogram_weights()

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)

        # calculate number of correct predictions
        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).float().sum()
        self.validation_step_outputs.append((num, correct))
        return num, correct

    def on_validation_epoch_end(self):
        # calculate and log top1 accuracy
        if self.validation_step_outputs:
            total_num = 0
            total_correct = 0
            for num, correct in self.validation_step_outputs:
                total_num += num
                total_correct += correct
            acc = total_correct / total_num
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=30.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
        return [optim], [scheduler]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# resnet = torchvision.models.resnet18()
# backbone = nn.Sequential(*list(resnet.children())[:-1])
# input_dim = 512
# instead of a resnet you can also use a vision transformer backbone as in the
# original paper (you might have to reduce the batch size in this case):
backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
input_dim = backbone.embed_dim
logger = TensorBoardLogger("tb_logs", name="my_model")
model = DINO(backbone, input_dim)
print(f"Model has {count_parameters(model):,} trainable parameters")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = DINOTransform()


# we ignore object detection annotations by setting target_transform to return 0
def target_transform(t):
    return 0

train_classifier_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)

train_dataset = torchvision.datasets.VOCDetection(
    "datasets/pascal_voc",
    download=True,
    transform=transform,
    target_transform=target_transform,
)

dataset_train_classifier = LightlyDataset(
    input_dir="datasets/cifar10/cifar10/train/", transform=train_classifier_transforms
)
dataset_test = LightlyDataset(input_dir="datasets/cifar10/cifar10/test", transform=test_transforms)

# train_size = int(0.8 * len(dataset))
# eval_size = len(dataset) - train_size
# train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
# eval_dataloader = torch.utils.data.DataLoader(
#     eval_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     drop_last=False,
#     num_workers=num_workers,
#     pin_memory=pin_memory,
# )


dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)


criterion = DINOLoss(
    output_dim=2048,
    warmup_teacher_temp_epochs=5,
)
# move loss to correct device because it also contains parameters
criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = epochs

import time
# st = time.perf_counter_ns()
# for epoch in range(epochs):
#     for batch in dataloader:
#         views = batch[0]
#         views = [view.to(device) for view in views]
#         global_views = views[:2]
# end = time.perf_counter_ns()
# initial_time = end - st

# with open("time_log_dino.txt", "a") as f:
#     f.write(f"Setting - Epoch: {epochs} #Workers: {num_workers} BS: {batch_size} pin: {pin_memory} --- Loading time: {initial_time} ns\n")

print("Starting Training")
st = time.perf_counter_ns()
#data_loading_times = []
#gpu_computation_times = []
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    #emit_nvtx=True,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    for epoch in range(epochs):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
        for batch in train_dataloader:
            #load_start = time.perf_counter_ns()
            views = batch[0]
            views = [view.to(device) for view in views]
            #load_end = time.perf_counter_ns()
            #data_loading_times.append(load_end - load_start)
            #gpu_start = time.perf_counter_ns()

            update_momentum(model.student_backbone, model.teacher_backbone, m=momentum_val)
            update_momentum(model.student_head, model.teacher_head, m=momentum_val)

            global_views = views[:2]
            teacher_out = [model.forward_teacher(view) for view in global_views]
            student_out = [model.forward(view) for view in views]
            loss = criterion(teacher_out, student_out, epoch=epoch)
            total_loss += loss.detach()
            loss.backward()
            # We only cancel gradients of student head.
            model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            optimizer.step()
            optimizer.zero_grad()
            #gpu_end = time.perf_counter_ns()
            #gpu_computation_times.append(gpu_end - gpu_start)
            prof.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

end = time.perf_counter_ns()
print(f"Entire Training Time: {end - st} ns")

# avg_data_loading_time = sum(data_loading_times) / len(data_loading_times)
# print(f"Average Data Loading Time per Batch: {avg_data_loading_time} ns")

# avg_gpu_computation_time = sum(gpu_computation_times) / len(gpu_computation_times)
# print(f"Average GPU Computation Time per Batch: {avg_gpu_computation_time} ns")

# print(prof.key_averages().table(
#     sort_by="self_cuda_time_total", row_limit=-1))
# with open("time_log_dino.txt", "a") as f:
#     f.write(f"Setting - Epoch: {epochs} #Workers: {num_workers} BS: {batch_size} pin: {pin_memory} --- Entire time: {end-st} ns\n")

# def evaluate_model(model, dataloader, device):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for batch in dataloader:
#             inputs, labels = batch
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#     accuracy = accuracy_score(all_labels, all_preds)
#     return accuracy

# accuracy = evaluate_model(model, eval_dataloader, device)
# print(f"Downstream Task Accuracy: {accuracy:.4f}")

model.eval()
classifier = Classifier(model.backbone)
trainer = pl.Trainer(max_epochs=epochs, devices=1, accelerator="gpu", logger=logger)
trainer.fit(classifier, dataloader_train_classifier, dataloader_test)

