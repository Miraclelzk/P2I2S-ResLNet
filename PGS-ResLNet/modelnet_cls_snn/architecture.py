import __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib_snn.dense import BasicConv, GraphConv2d, ResDynBlock2d, DenseDynBlock2d, DilatedKnnGraph, PlainDynBlock2d
from gcn_lib_snn.dense import BasicConv_snn, ResDynBlock2d_snn,GraphConv2d_snn

from spikingjelly.activation_based import layer as spiking_layer

class DeepGCN_SNN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN_SNN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        # act = opt.act
        act = 'if'
        norm = opt.norm
        bias = opt.bias
        knn = 'matrix'  # implement knn using matrix multiplication
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        c_growth = channels
        emb_dims = opt.emb_dims
        self.n_blocks = opt.n_blocks

        self.T = opt.T

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d_snn(opt.in_channels, channels,self.T, conv, act, norm, bias=False)

        if opt.block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon, knn)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks-1)) * self.n_blocks // 2)

        elif opt.block.lower() == 'res':

            if opt.use_dilation:
                self.backbone = Seq(*[ResDynBlock2d_snn(channels,self.T, k, i + 1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn)
                                      for i in range(self.n_blocks - 1)])
            else:
                self.backbone = Seq(*[ResDynBlock2d_snn(channels,self.T, k,  1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn)
                                      for _ in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        else:
            # Plain GCN. No dilation, no stochastic, no residual connections
            stochastic = False

            self.backbone = Seq(*[PlainDynBlock2d(channels, k, 1, conv, act, norm,
                                                  bias, stochastic, epsilon, knn)
                                  for i in range(self.n_blocks - 1)])

            fusion_dims = int(channels+c_growth*(self.n_blocks-1))

        self.fusion_block = BasicConv([fusion_dims, emb_dims], 'leakyrelu', norm, bias=False)
        self.prediction = Seq(*[BasicConv([emb_dims * 2, 512], 'leakyrelu', norm, drop=opt.dropout),
                                BasicConv([512, 256], 'leakyrelu', norm, drop=opt.dropout),
                                BasicConv([256, opt.n_classes], None, None)])
        
        # self.fusion_block = BasicConv_snn([fusion_dims, emb_dims], 'IF', norm, bias=False)
        # self.prediction = Seq(*[BasicConv_snn([emb_dims * 2, 512], 'IF', norm, drop=opt.dropout),
        #                         BasicConv_snn([512, 256], 'IF', norm, drop=opt.dropout),
        #                         BasicConv_snn([256, opt.n_classes], None, None)])
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):

        if self.T != 1:
            inputs = inputs.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            inputs = inputs.flatten(0, 1)

        feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]
 
        # GCN Bsckbone Block
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))
        
        feats = torch.cat(feats, dim=-3)

        if self.T != 1:
            feats = feats.mean(0)

        # Fusion Block
        fusion = self.fusion_block(feats)

        x1 = F.adaptive_max_pool2d(fusion, 1)
        x2 = F.adaptive_avg_pool2d(fusion, 1)

        out = torch.cat((x1, x2), dim=1)

        # MLP Prediction Block
        out = self.prediction(out).squeeze(-1).squeeze(-1)

        return out
    
