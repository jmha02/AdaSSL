import torch
import copy
import pytorch_lightning as pl
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.models.modules import DINOProjectionHead
from lightly.utils.scheduler import cosine_schedule
from lightly.loss import DINOLoss

from peft import LoraConfig, AdaLoraConfig, IA3Config, get_peft_model

class DINO(pl.LightningModule):
    def __init__(self, backbone, input_dim, lora_rank=4, lora_alpha=1.0, epochs=10):
        super().__init__()
        self.epochs = epochs
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)

        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(
            output_dim=2048, 
            warmup_teacher_temp_epochs=5,
        )

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        views = batch[0]
        global_views = views[:2]

        momentum_val = cosine_schedule(self.current_epoch, self.epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum_val)
        update_momentum(self.student_head, self.teacher_head, m=momentum_val)

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student_backbone.parameters(), lr=5e-4, weight_decay=1e-4)
        return optimizer

class DINO_LoRA(pl.LightningModule):
    def __init__(self, backbone, input_dim, lora_rank=4, lora_alpha=1.0, epochs=10):
        super().__init__()
        self.epochs = epochs
        self.student_backbone = backbone
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            bias="none",
        )
        self.student_backbone = get_peft_model(self.student_backbone, lora_config)
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)

        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(
            output_dim=2048, 
            warmup_teacher_temp_epochs=5,
        )

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        views = batch[0]
        global_views = views[:2]

        momentum_val = cosine_schedule(self.current_epoch, self.epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum_val)
        update_momentum(self.student_head, self.teacher_head, m=momentum_val)

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student_backbone.parameters(), lr=5e-4, weight_decay=1e-4)
        return optimizer


class DINO_AdaLoRA(pl.LightningModule):
    def __init__(self, backbone, input_dim, lora_rank=4, lora_alpha=1.0, epochs=10):
        super().__init__()
        self.epochs = epochs
        self.student_backbone = backbone
        lora_config = AdaLoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            bias="none",
        )
        self.student_backbone = get_peft_model(self.student_backbone, lora_config)
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)

        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(
            output_dim=2048, 
            warmup_teacher_temp_epochs=5,
        )

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        views = batch[0]
        global_views = views[:2]

        momentum_val = cosine_schedule(self.current_epoch, self.epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum_val)
        update_momentum(self.student_head, self.teacher_head, m=momentum_val)

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student_backbone.parameters(), lr=5e-4, weight_decay=1e-4)
        return optimizer


class DINO_IA3(pl.LightningModule):
    def __init__(self, backbone, input_dim, lora_rank=4, lora_alpha=1.0, epochs=10):
        super().__init__()
        self.epochs = epochs
        self.student_backbone = backbone
        lora_config = IA3Config(
            target_modules=["qkv", "fc1", "fc2"],
            feedforward_modules=["fc1", "fc2"],
        )
        self.student_backbone = get_peft_model(self.student_backbone, lora_config)
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)

        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(
            output_dim=2048, 
            warmup_teacher_temp_epochs=5,
        )

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        views = batch[0]
        global_views = views[:2]

        momentum_val = cosine_schedule(self.current_epoch, self.epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum_val)
        update_momentum(self.student_head, self.teacher_head, m=momentum_val)

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student_backbone.parameters(), lr=5e-4, weight_decay=1e-4)
        return optimizer

class DINO_SparseLoRA_Tensor(pl.LightningModule):
    def __init__(self, backbone, input_dim, lora_rank=4, lora_alpha=1.0, epochs=10, sparse_threshold=0.01, sparse_decay=0.99):
        super().__init__()
        self.epochs = epochs
        self.sparse_threshold = sparse_threshold
        self.sparse_decay = sparse_decay

        self.student_backbone = backbone
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            bias="none",
        )
        self.student_backbone = get_peft_model(self.student_backbone, self.lora_config)
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)

        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        # DINO Loss
        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

        self.layer_importance = {name: 0 for name, _ in self.student_backbone.named_parameters()}
        self.tensor_importance = {}

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        views = batch[0]
        global_views = views[:2]

        momentum_val = cosine_schedule(self.current_epoch, self.epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum_val)
        update_momentum(self.student_head, self.teacher_head, m=momentum_val)

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]

        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss", loss)

        self.compute_importance(loss)
        self.apply_sparse_lora_tensor()

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.student_backbone.parameters()), 
                                      lr=5e-4, weight_decay=1e-4)
        return optimizer

    def compute_importance(self, loss):
        # loss_change = abs(loss.detach().cpu().item() - getattr(self, "prev_loss", 0))
        # self.prev_loss = loss.detach().cpu().item()
        
        current_loss = loss.detach().cpu().item()
        prev_loss = getattr(self, "prev_loss", current_loss)
        loss_change = abs(current_loss - prev_loss)
        self.prev_loss = current_loss
        decay = 0.9

        for name, param in self.student_backbone.named_parameters():
            if param.grad is not None:
                importance_score = param.grad.abs().mean().item()
                importance_with_loss = loss_change * importance_score
                self.layer_importance[name] = decay * self.layer_importance.get(name, 0) + (1 - decay) * importance_with_loss
                
                tensor_name = f"{name}_tensor"
                if tensor_name not in self.tensor_importance:
                    self.tensor_importance[tensor_name] = torch.zeros_like(param)
                self.tensor_importance[tensor_name] = decay * self.tensor_importance[tensor_name] + (1 - decay) * param.grad.abs()
                # with torch.no_grad():
                #     tensor_name = f"{name}_tensor"
                #     if tensor_name not in self.tensor_importance:
                #         self.tensor_importance[tensor_name] = torch.zeros_like(param)
                #     self.tensor_importance[tensor_name] += param.grad.abs()

    def apply_sparse_lora_tensor(self):
        new_lora_targets = []
        for name, param in self.student_backbone.named_parameters():
            if self.layer_importance[name] < self.sparse_threshold:
                param.requires_grad = False
            else:
                tensor_name = f"{name}_tensor"
                if tensor_name in self.tensor_importance:
                    tensor_importance_score = self.tensor_importance[tensor_name].mean().item()
                    if tensor_importance_score < self.sparse_threshold:
                        param.requires_grad = False

                if "qkv" in name:
                    new_lora_targets.append(name.split(".")[0])

        if new_lora_targets:
            self.lora_config.target_modules = list(set(new_lora_targets))
            self.student_backbone = get_peft_model(self.student_backbone, self.lora_config)
        
        self.sparse_threshold *= self.sparse_decay