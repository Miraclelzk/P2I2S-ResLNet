import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from spikingjelly.activation_based import neuron, functional, surrogate, layer

class VOXELSNN(nn.Module):
    def __init__(self, cfg, is_test=False):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.timestep

        if cfg.base_model_variant == 'ms_resnet18':
            self.base_model = create_model(cfg.base_model_variant)
            functional.set_step_mode( self.base_model, step_mode='m')
        else:

            if 'sew_resnet' in cfg.base_model_variant:
                self.base_model = create_model(cfg.base_model_variant, cnf_=cfg.cnf)
            elif 'spiking_resnet' in cfg.base_model_variant:
                self.base_model = create_model(cfg.base_model_variant)

            self.base_model.num_features = self.base_model.fc.in_features
            
            if cfg.head_type == 'mlp':
                from models.layers.head import MLPHead
                cls_head = MLPHead(self.base_model.num_features, cfg.classes, cfg.mlp_mid_channels, cfg.mlp_dropout_ratio)
            elif cfg.head_type == 'linear':
                cls_head = layer.Linear(self.base_model.num_features, cfg.classes)
            else:
                raise ValueError('cfg.head_type is not defined!')
            
            self.base_model.fc = cls_head
            if self.T != 1:
                functional.set_step_mode( self.base_model, step_mode='m')
                functional.set_backend(self.base_model, 'cupy', neuron.IFNode)
            else:
                functional.set_step_mode( self.base_model, step_mode='s')

        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, pred, gt, smoothing=True):
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss_cls = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss_cls = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss_cls, acc * 100

    def forward(self, voxel):
        if self.T != 1:
            voxel = voxel.unsqueeze(0).repeat(self.T, 1, 1, 1, 1, 1)
        out = self.base_model(voxel)
        if self.T != 1:
            out = out.mean(0)
        return out
