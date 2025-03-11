import torch
import torchvision
import argparse
from models import simclr, dino, classifier, byol, mae, moco, simsiam

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.profilers import PyTorchProfiler
import time
import os
#===--- Parsing Arguments ---===#
parser = argparse.ArgumentParser(description="Training SSL on CIFAR100")
parser.add_argument("--model", type=str, required=True, help="SSL Model Name (e.g. SimCLR / DINO / DINO_LoRA / DINO_AdaLoRA / DINO_IA3)")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")
args = parser.parse_args()
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

if args.model == 'SimCLR':
    model_path = os.path.join(save_dir, "SimCLR.pth")
    transform = transforms.SimCLRTransform(input_size=32, cj_prob=0.5)
    model = simclr.SimCLR()
elif args.model == 'DINO':
    model_path = os.path.join(save_dir, "DINO.pth")
    transform = transforms.DINOTransform()
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
    input_dim = backbone.embed_dim
    model = dino.DINO(backbone, input_dim, lora_rank=8, lora_alpha=1.0, epochs=args.epochs)
elif args.model == 'DINO_LoRA':
    model_path = os.path.join(save_dir, "DINO_LoRA.pth")
    transform = transforms.DINOTransform()
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
    input_dim = backbone.embed_dim
    model = dino.DINO_LoRA(backbone, input_dim, lora_rank=8, lora_alpha=1.0, epochs=args.epochs)
elif args.model == 'DINO_AdaLoRA':
    model_path = os.path.join(save_dir, "DINO_AdaLoRA.pth")
    transform = transforms.DINOTransform()
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
    input_dim = backbone.embed_dim
    model = dino.DINO_AdaLoRA(backbone, input_dim, lora_rank=8, lora_alpha=1.0, epochs=args.epochs)
elif args.model == 'DINO_IA3':
    model_path = os.path.join(save_dir, "DINO_IA3.pth")
    transform = transforms.DINOTransform()
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
    input_dim = backbone.embed_dim
    model = dino.DINO_IA3(backbone, input_dim, lora_rank=8, lora_alpha=1.0, epochs=args.epochs)
# elif args.model == 'DINO_SparseUpdate':
#     model_path = os.path.join(save_dir, "DINO_SparseUpdate.pth")
#     transform = transforms.DINOTransform()
#     backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
#     input_dim = backbone.embed_dim
#     model = dino.DINO_SparseUpdate(backbone, input_dim, lora_rank=8, lora_alpha=1.0, epochs=args.epochs)
# elif args.model == 'DINO_SparseLoRA':
#     model_path = os.path.join(save_dir, "DINO_SparseLoRA.pth")
#     transform = transforms.DINOTransform()
#     backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
#     input_dim = backbone.embed_dim
#     model = dino.DINO_SparseLoRA(backbone, input_dim, lora_rank=8, lora_alpha=1.0, epochs=args.epochs)
elif args.model == 'DINO_SparseLoRA_Tensor':
    model_path = os.path.join(save_dir, "DINO_SparseLoRA_Tensor.pth")
    transform = transforms.DINOTransform()
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
    input_dim = backbone.embed_dim
    model = dino.DINO_SparseLoRA_Tensor(backbone, input_dim, lora_rank=8, lora_alpha=1.0, epochs=args.epochs)
elif args.model == 'MoCo':
    pass
elif args.model == 'BYOL':
    pass
elif args.model == 'SimSiam':
    pass
elif args.model == 'MAE':
    pass
else:
    raise ValueError("Invalid Model Name")

#===- Self-Supervised Learning -===#
cifar100_dataset = torchvision.datasets.CIFAR100(
    root='./datasets',
    train=True,
    download=True,
    transform=transform
)

dataset = LightlyDataset.from_torch_dataset(cifar100_dataset)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
)

profiler = PyTorchProfiler(
    dirpath="./log",
    filename="profiler_output",
    emit_nvtx=True,
    record_functions=True,
)
# if os.path.exists(model_path):
#     model = model.load_state_dict(torch.load(model_path))
#     print(f"Model loaded from {model_path}")
# else:
trainer = Trainer(max_epochs=args.epochs, devices=1, accelerator="gpu", profiler=profiler)
st = time.perf_counter_ns()
trainer.fit(model, dataloader)
end = time.perf_counter_ns()
print(f"===--- Training Time: {(end - st) / 1e9:.2f}s ---===")
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")

#===--- Downstream Task ---===#

classifier_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
])

train_dataset = torchvision.datasets.CIFAR100(
    root="./datasets",
    train=True,
    download=True,
    transform=classifier_transform
)
test_dataset = torchvision.datasets.CIFAR100(
    root="./datasets",
    train=False,
    download=True,
    transform=classifier_transform
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

model.eval()
if args.model == 'SimCLR':
    classifier_model = classifier.SimCLR_Classifier(model)
else:
    classifier_model = classifier.DINO_Classifier(model)

trainer = Trainer(max_epochs=50, devices=1, accelerator="gpu")
trainer.fit(classifier_model, train_dataloader)

from tqdm import tqdm
#===--- Evaluation - Linear Probing ---===# 
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating Linear Probing"):
            x, y = x.to(model.device), y.to(model.device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total * 100
    return accuracy

test_accuracy = evaluate(classifier_model, test_dataloader)
print(f"Linear Probing Test Accuracy: {test_accuracy:.2f}%")

#===--- Evaluation - KNN Accuracy ---===#
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
def extract_features(model, dataloader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting Features for KNN"):
            x, y = x.to(model.device), y.to(model.device)
            if args.model == 'SimCLR':
                feature = model.backbone(x).flatten(start_dim=1)
            else:
                feature = model.student_backbone(x).flatten(start_dim=1)
            features.append(feature.cpu().numpy())
            labels.append(y.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

train_features, train_labels = extract_features(model, train_dataloader)
test_features, test_labels = extract_features(model, test_dataloader)

knn = KNeighborsClassifier(n_neighbors=20, metric="cosine")
knn.fit(train_features, train_labels)
test_preds = knn.predict(test_features)

knn_accuracy = (test_preds == test_labels).mean() * 100
print(f"KNN Test Accuracy: {knn_accuracy:.2f}%")