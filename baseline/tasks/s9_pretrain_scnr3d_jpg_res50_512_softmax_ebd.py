import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from datetime import date
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils import data
import torch.multiprocessing
from torchvision import transforms, models
import numpy as np
from pytorch_metric_learning import losses as pml_losses
from genetic_marker_dataset import PretrainImageDataset
from embedding import SoftmaxEmbeddingNet
from train import Trainer 
import util_hooks
import metric
from hparam import HParam
torch.multiprocessing.set_sharing_strategy('file_system')

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', class_by='scan_date') -> None:
        super().__init__(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.class_by = class_by

    def forward(self, input, target):
        return super().forward(input, target[self.class_by].cuda())

def log_epoch_acc_hook(trainer):
    trainer.logger.run['epoch/val_recall_at_5'].log(trainer.epoch_val_acc, timestamp=trainer.logger.epoch_counter)

def acc_func(vectors, labels):
    if type(vectors) == torch.Tensor:
        device = vectors.device
    else:
        device = None
    predict_cls = torch.argmax(F.softmax(vectors, dim=1), dim=1).type(labels[dataset_class_by].dtype)
    acc = torch.mean(labels[dataset_class_by].to(device).eq(predict_cls).type(torch.float))
    return acc  

exp_name = os.path.splitext(os.path.basename(__file__))[0]
resume_dict = None

hps = HParam(img_crop_size = 512,
             img_resize = 512,
             resnet_n = 50,
             init_lr = 1e-2,
             batch_size = 30,
             sampler_date_shuffle=True)

print('Experiment name: {}'.format(exp_name))
print('Load dataset')
dataset_class_by = 'plot_cls'
image_transform = transforms.Compose([transforms.RandomCrop(hps.img_crop_size, pad_if_needed=True),
                                      transforms.Resize(hps.img_resize),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor()])
val_image_transform = transforms.Compose([transforms.RandomCrop(hps.img_crop_size, pad_if_needed=True),
                                      transforms.Resize(hps.img_resize),
                                      transforms.ToTensor()])
train_dataset = PretrainImageDataset('/data/shared/nips_pretrain/', sensor='3d', transform=image_transform)
val_dataset = PretrainImageDataset('/data/shared/nips_pretrain/', sensor='3d', train=False, transform=val_image_transform)
num_class = len(np.unique(train_dataset.metadata_df[dataset_class_by]))
dataloader = data.DataLoader(train_dataset, batch_size=hps.batch_size, shuffle=True, num_workers=22)
val_dataloader = data.DataLoader(val_dataset, batch_size=125, shuffle=False, num_workers=22)

print('Init model')

model = SoftmaxEmbeddingNet(num_classes=num_class)

loss_func = CrossEntropyLoss(class_by=dataset_class_by)
model.cuda()
optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=hps.init_lr)
scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
t = Trainer(model, dataloader, loss_func, optimizer, scheduler, n_epochs=120, train_name=exp_name,
            acc_func=acc_func,
            logger='neptune', log_dir='zeyu/reverse-pheno', log_hparam=hps.__dict__)

hooks = [util_hooks.construct_test_hook(val_dataloader),
         util_hooks.construct_metric_hook(metric.recall_at_k_by_cultivar),
         log_epoch_acc_hook]
for h in hooks:
    t.add_epoch_hook(h)
 
t.train()