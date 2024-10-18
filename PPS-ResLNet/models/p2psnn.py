import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from models.layers.encoder import ProjEnc
from spikingjelly.activation_based import neuron, functional, surrogate, layer

class P2PSNN(nn.Module):
    def __init__(self, cfg, is_test=False):
        super().__init__()
        self.cfg = cfg
        self.enc = ProjEnc(cfg)
        self.T = cfg.timestep

        if 'ms_resnet' in cfg.base_model_variant :
            self.base_model = create_model(cfg.base_model_variant)
            # functional.set_step_mode( self.base_model, step_mode='m')
        else:
            if 'sew_resnet18' in cfg.base_model_variant:
                if is_test:
                    self.base_model = create_model(cfg.base_model_variant, cnf_=cfg.cnf)
                else:
                    self.base_model = create_model(cfg.base_model_variant, cnf_=cfg.cnf, checkpoint_path=cfg.checkpoint_path)
                    
            elif 'spiking_resnet' in cfg.base_model_variant:
                if is_test:
                    self.base_model = create_model(cfg.base_model_variant)
                else:
                    self.base_model = create_model(cfg.base_model_variant, checkpoint_path=cfg.checkpoint_path)

            self.base_model.num_features = self.base_model.fc.in_features
            
            if cfg.head_type == 'mlp':
                from models.layers.head import MLPHead
                cls_head = MLPHead(self.base_model.num_features, cfg.classes, cfg.mlp_mid_channels, cfg.mlp_dropout_ratio)
            elif cfg.head_type == 'linear':
                cls_head = nn.Linear(self.base_model.num_features, cfg.classes)
            else:
                raise ValueError('cfg.head_type is not defined!')
            

            self.base_model.fc = cls_head

            if self.T != 1:
                functional.set_step_mode( self.base_model, step_mode='m')
                functional.set_backend(self.base_model, 'cupy', neuron.IFNode)
            else:
                functional.set_step_mode( self.base_model, step_mode='s')


        self.loss_ce = nn.CrossEntropyLoss()

    def _fix_weight(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        # learnable cls head parameters

        if 'resnet' in self.cfg.base_model_variant:
            for param in self.base_model.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.base_model.head.parameters():
                param.requires_grad = True

        # flexible learnable parameters
        if self.cfg.update_type is not None:
            for name, param in self.base_model.named_parameters():
                if self.cfg.update_type in name:
                    param.requires_grad = True
            print('Learnable {} parameters!'.format(self.cfg.update_type))

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

    def forward(self, pc, original_pc):
        img, _ = self.enc(original_pc, pc)
        if self.T != 1:
            img = img.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.base_model(img)
        if self.T != 1:
            out = out.mean(0)
        return out
