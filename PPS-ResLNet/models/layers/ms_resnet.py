import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer
from timm.models.registry import register_model
# Model for RM-ResNet

thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.25  # decay constants

time_window = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        temp = temp / (2 * lens)
        return grad_input * temp.float()


act_fun = ActFun.apply
# membrane potential update


class mem_update_lif(nn.Module):

    def __init__(self):
        super(mem_update_lif, self).__init__()

    def forward(self, x):
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                mem = mem_old * decay * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            spike = act_fun(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output

class mem_update_if(nn.Module):

    def __init__(self):
        super(mem_update_if, self).__init__()

    def forward(self, x):
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                mem = mem_old * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            spike = act_fun(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


class batch_norm_2d(nn.Module):
    """TDBN"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class batch_norm_2d1(nn.Module):
    """TDBN-Zero init"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0)
            nn.init.zeros_(self.bias)


class Snn_Conv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 marker='b'):
        super(Snn_Conv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        weight = self.weight
        h = (input.size()[3] - self.kernel_size[0] +
             2 * self.padding[0]) // self.stride[0] + 1
        w = (input.size()[4] - self.kernel_size[0] +
             2 * self.padding[0]) // self.stride[0] + 1
        c1 = torch.zeros(time_window,
                         input.size()[1],
                         self.out_channels,
                         h,
                         w,
                         device=input.device)
        for i in range(time_window):
            c1[i] = F.conv2d(input[i], weight, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        return c1


######################################################################################################################



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            mem_update_lif(),
            Snn_Conv2d(in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=stride,
                       padding=1,
                       bias=False),
            batch_norm_2d(out_channels),
            mem_update_lif(),
            Snn_Conv2d(out_channels,
                       out_channels * BasicBlock.expansion,
                       kernel_size=3,
                       padding=1,
                       bias=False),
            batch_norm_2d1(out_channels * BasicBlock.expansion),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels,
                           out_channels * BasicBlock.expansion,
                           kernel_size=1,
                           stride=stride,
                           bias=False),
                batch_norm_2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return (self.residual_function(x) + self.shortcut(x))

class BasicBlock_104(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            mem_update_lif(),
            Snn_Conv2d(in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=stride,
                       padding=1,
                       bias=False),
            batch_norm_2d(out_channels),
            mem_update_lif(),
            Snn_Conv2d(out_channels,
                       out_channels * BasicBlock_104.expansion,
                       kernel_size=3,
                       padding=1,
                       bias=False),
            batch_norm_2d1(out_channels * BasicBlock_104.expansion),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock_104.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
                Snn_Conv2d(in_channels,
                           out_channels * BasicBlock_104.expansion,
                           kernel_size=1,
                           stride=1,
                           bias=False),
                batch_norm_2d(out_channels * BasicBlock_104.expansion),
            )

    def forward(self, x):
        return (self.residual_function(x) + self.shortcut(x))


class ResNet_104(nn.Module):
    # Channel:
    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()
        k = 1
        self.in_channels = 64 * k
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3, 64 * k, kernel_size=3, padding=1, stride=2),
            Snn_Conv2d(64 * k, 64 * k, kernel_size=3, padding=1, stride=1),
            Snn_Conv2d(64 * k, 64 * k, kernel_size=3, padding=1, stride=1),
            batch_norm_2d(64 * k),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.mem_update = mem_update_lif()
        self.conv2_x = self._make_layer(block, 64 * k, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128 * k, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256 * k, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512 * k, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion * k, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def _make_layer(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        input = torch.zeros(time_window,
                            x.size()[0],
                            3,
                            x.size()[2],
                            x.size()[3],
                            device=device)
        for i in range(time_window):
            input[i] = x
        output = self.conv1(input)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.mem_update(output)
        output = F.adaptive_avg_pool3d(output, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0) / output.size()[0]
        output = self.dropout(output)
        output = self.fc(output)
        return output




class ResNet_origin_18(nn.Module):
    # Channel:
    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()
        k = 1
        self.in_channels = 64 * k
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3,
                       64 * k,
                       kernel_size=7,
                       padding=3,
                       bias=False,
                       stride=2),
            batch_norm_2d(64 * k),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.mem_update = mem_update_lif()
        self.conv2_x = self._make_layer(block, 64 * k, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128 * k, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256 * k, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512 * k, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion * k, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        input = torch.zeros(time_window,
                            x.size()[0],
                            3,
                            x.size()[2],
                            x.size()[3],
                            device=device)
        for i in range(time_window):
            input[i] = x
        output = self.conv1(input)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.mem_update(output)
        output = F.adaptive_avg_pool3d(output, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0) / output.size()[0]
        output = self.fc(output)
        return output

@register_model
def ms_resnet18(pretrained=False, **kwargs):
    num_classes = 40
    model = ResNet_origin_18(BasicBlock, [2, 2, 2, 2], num_classes)
    return model

@register_model
def ms_resnet34(pretrained=False, **kwargs):
    num_classes = 40
    return ResNet_origin_18(BasicBlock, [3, 4, 6, 3], num_classes)

@register_model
def ms_resnet104(pretrained=False, **kwargs):
    num_classes = 40
    return ResNet_104(BasicBlock_104, [3, 8, 32, 8], num_classes)
