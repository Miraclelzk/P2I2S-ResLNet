import torch
from torch import nn
from .torch_nn import BasicConv, BasicConv_snn, batched_index_select
from .torch_edge import DenseDilatedKnnGraph, DilatedKnnGraph






class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        return self.nn(torch.cat([x, x_j], dim=1))


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DynConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        if knn == 'matrix':
            self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        else:
            self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)

    
class PlainDynBlock2d(nn.Module):
    """
    Plain Dynamic graph convolution block
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, knn='matrix'):
        super(PlainDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index)
    
    
class ResDynBlock2d(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, knn='matrix', res_scale=1):
        super(ResDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)
        self.res_scale = res_scale

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index) + x*self.res_scale


class DenseDynBlock2d(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels=64,  kernel_size=9, dilation=1, conv='edge',
                 act='relu', norm=None,bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DenseDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1)












##############################
#    SNN layers
##############################

class MRConv2d_snn(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='if', norm=None, bias=True):
        super(MRConv2d_snn, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        return self.nn(torch.cat([x, x_j], dim=1))


class EdgeConv2d_snn(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, T, act='if', norm=None, bias=True):
        super(EdgeConv2d_snn, self).__init__()
        self.nn = BasicConv_snn([in_channels * 2, out_channels], act, norm, bias)
        self.T = T

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])

        out = torch.cat([x_i, x_j - x_i], dim=1)
        if self.T != 1:
            TB,FF,NN,LL = out.shape
            out = out.reshape(self.T,-1,FF,NN,LL)

        out = self.nn(out)

        max_value, _ = torch.max(out, -1, keepdim=True)

        return max_value


class GraphConv2d_snn(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, T, conv='edge', act='if', norm=None, bias=True):
        super(GraphConv2d_snn, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d_snn(in_channels, out_channels, T, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d_snn(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv2d_snn(GraphConv2d_snn):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels,T, kernel_size=9, dilation=1, conv='edge', act='if',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DynConv2d_snn, self).__init__(in_channels, out_channels,T, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.T = T
        if knn == 'matrix':
            self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        else:
            self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, edge_index=None):
        if self.T != 1:
            x = x.flatten(0, 1)
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d_snn, self).forward(x, edge_index)

    
class PlainDynBlock2d_snn(nn.Module):
    """
    Plain Dynamic graph convolution block
    """
    def __init__(self, in_channels,T, kernel_size=9, dilation=1, conv='edge', act='if', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, knn='matrix'):
        super(PlainDynBlock2d_snn, self).__init__()
        self.body = DynConv2d_snn(in_channels, in_channels,T, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index)
    
    
class ResDynBlock2d_snn(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, in_channels, T, kernel_size=9, dilation=1, conv='edge', act='if', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, knn='matrix', res_scale=1):
        super(ResDynBlock2d_snn, self).__init__()
        self.body = DynConv2d_snn(in_channels, in_channels,T, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)
        self.res_scale = res_scale

    def forward(self, x, edge_index=None):


        return self.body(x, edge_index) + x*self.res_scale


class DenseDynBlock2d_snn(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels,T, out_channels=64,  kernel_size=9, dilation=1, conv='edge',
                 act='if', norm=None,bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DenseDynBlock2d_snn, self).__init__()
        self.body = DynConv2d_snn(in_channels, out_channels,T, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1)
