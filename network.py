import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class PadConv2d(nn.Module):
    def __init__(self, *args, padding_mode='constant', **kwargs):
        super(PadConv2d, self).__init__()
        self.pad = args[4]
        kwargs_new = kwargs
        kwargs['in_channels'] = args[0]
        kwargs['out_channels'] = args[1]
        kwargs['kernel_size'] = args[2]
        kwargs['stride'] = args[3]
        kwargs['padding'] = 0
        self.conv = nn.Conv2d(**kwargs_new)
        self.padding_mode = padding_mode

    def forward(self, x):
        x = F.pad(x, [self.pad for _ in range(4)], mode=self.padding_mode)
        return self.conv(x)

    def __repr__(self):
        return str(self.conv) + '\tpadding_mode: {}'.format(self.padding_mode)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(F.relu6(x.add_(3), self.inplace)).div_(6)
        #return x * (F.relu6(x+3, self.inplace)/6)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x.add_(3), self.inplace).div_(6)
        #return F.relu6(x+3, self.inplace)/6


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                m.weight.data += 1e-4 * torch.randn(*m.weight.shape)
                init.constant_(m.bias.data, 0.0)
                m.bias.requires_grad = False


def make_layer(block, cfg):
    layers = []
    for idx in range(len(cfg)):
        bc = cfg[idx]
        layers.append(block(bc=bc))
    return nn.Sequential(*layers)

# ----
# Se Layer: refer to https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
# TODO: weight initialization ?
# TODO relu / h_swish
# ----


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, activation='relu'):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if activation == 'relu':
            self.activation = nn.ReLU
        else:
            self.activation = h_swish

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            self.activation(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            h_sigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze()
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ----
# h-swish: refer to https://github.com/leaderj1001/MobileNetV3-Pytorch/blob/master/model.py
# ----


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, ic, bc, oc, activation='relu', bias=False, use_bn=True):
        super(ResidualBlock_noBN, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(ic, bc, 3, 1, 1, bias=bias)

        if self.use_bn:
            self.bn = nn.BatchNorm2d(bc)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'h-swish':
            self.activation = h_swish(inplace=True)
        else:
            print('activation function must be relu or h-swish')

        self.conv2 = nn.Conv2d(bc, oc, 3, 1, 1, bias=bias)

        if use_bn:  # initialization
            initialize_weights([self.conv1, self.bn, self.conv2], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2], 0.1)

    def naive_bn(self, x):
        a = self.bn.weight.view(1, -1, 1, 1)
        return x.mul_(a)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        if self.use_bn:
            x = self.naive_bn(x)
        x = self.activation(x)
        out = self.conv2(x)
        return identity + out


class SE_ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+---SE-->
     |________________|
    '''

    def __init__(self, ic, bc, oc, activation='relu', bias=False, use_dilation=False, padding_mode='constant', use_bn=True):
        super(SE_ResidualBlock_noBN, self).__init__()

        self.conv1 = nn.Conv2d(ic, bc, 3, 1, 1, bias=bias, padding_mode=padding_mode)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(bc)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'h-swish':
            self.activation = h_swish(inplace=True)
        else:
            print('activation function must be relu or h-swish')

        if use_dilation:
            self.conv2 = nn.Conv2d(bc, oc, 3, 1, 2, bias=bias, dilation=2, padding_mode=padding_mode)
        else:
            self.conv2 = nn.Conv2d(bc, oc, 3, 1, 1, bias=bias, padding_mode=padding_mode)

        self.se = SELayer(oc, reduction=16, activation=activation)
        if use_bn:
            # initialization
            initialize_weights([self.conv1, self.bn, self.conv2], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2], 0.1)

    def naive_bn(self, x):
        a = self.bn.weight.view(1, -1, 1, 1)
        return x.mul_(a)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.naive_bn(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.se(out)  # se layers
        return identity + out


class MSRResNetSlim(nn.Module):
    ''' Slim SRResNet'''
    ''' benchmark: nf = 64, nb = 16, activation = relu , mode  = benchmark'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, upscale=4, activation='h-swish', mode='se', use_y=False, res_location='last', bias=False,
                 interp_type='bicubic', use_reflect_pad=True, use_dilation=False, cfg=None, use_bn=False):
        super(MSRResNetSlim, self).__init__()
        self.use_y = use_y
        self.res_location = res_location
        self.interp_type = interp_type
        if use_y:
            in_nc = 1
            out_nc = 1
        if use_reflect_pad:
            padding_mode = 'reflect'
        else:
            padding_mode = 'constant'

        if activation not in ['relu', 'h-swish']:
            print('activation must be \'relu\' or \'h-swish\'')

        self.upscale = upscale

        self.use_bn = use_bn
        if cfg is None:
            cfg = [64]*nb

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=bias, padding_mode=padding_mode)

        if mode == 'se':
            block = functools.partial(
                    SE_ResidualBlock_noBN, ic=nf, oc=nf, activation=activation, bias=bias, use_dilation=use_dilation, padding_mode=padding_mode, use_bn=self.use_bn)
        elif mode == 'benchmark':
            block = functools.partial(
                ResidualBlock_noBN, ic=nf, oc=nf, activation=activation, bias=bias, use_bn=self.use_bn)
        else:
            print('mode must be \'benchmark\' or \'se\'')

        self.recon_trunk = make_layer(block, cfg)

        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=bias, padding_mode=padding_mode)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=bias, padding_mode=padding_mode)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias, padding_mode=padding_mode)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=bias, padding_mode=padding_mode)

        # activation function

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([self.conv_first, self.upconv1, self.upconv2,
                            self.HRconv, self.conv_last], 0.1)
        self.rgb2ypbpr = torch.FloatTensor([0.299, -0.14713, 0.615, 0.587, -0.28886, -0.51499, 0.114, 0.436, -0.10001]).view(3, 3)
        self.ypbpr2rgb = torch.FloatTensor([1., 1., 1., 0., -0.39465, 2.03211, 1.13983, -0.58060, 0.]).view(3, 3)

    def rgb2ypbpr_transform(self, x):
        rgb2ypbpr = self.rgb2ypbpr.to(x.device)
        return x.permute(0, 2, 3, 1).matmul(rgb2ypbpr).permute(0, 3, 1, 2)

    def ypbpr2rgb_transform(self, x):
        ypbpr2rgb = self.ypbpr2rgb.to(x.device)
        return x.permute(0, 2, 3, 1).matmul(ypbpr2rgb).permute(0, 3, 1, 2)

    def forward(self, x):
        if self.use_y:
            y = self.rgb2ypbpr_transform(x)
            x, color = y[:, 0:1], y[:, 1:]
            color = F.interpolate(color, scale_factor=4, mode=self.interp_type)
        fea = self.activation(self.conv_first(x))
        out = self.recon_trunk(fea)
        if self.res_location == 'mid':
            out += fea

        out = self.activation(self.pixel_shuffle(self.upconv1(out)))
        out = self.activation(self.pixel_shuffle(self.upconv2(out)))

        out = self.conv_last(self.activation(self.HRconv(out)))
        if self.res_location == 'last':
            base = F.interpolate(x, scale_factor=self.upscale,
                                 mode=self.interp_type)
            out += base

        if self.use_y:
            out = torch.cat([out, color], dim=1)
            out = self.ypbpr2rgb_transform(out)
        return out
