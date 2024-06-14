import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from pretrained import pretrained_model
from torch_nn import act_layer
from torch_vertex import Grapher

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 10, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vig_224_gelu': _cfg(
        # mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        mean = (0.4914, 0.4822, 0.4465), std = (0.247, 0.243, 0.261),
        # mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    ),
    'vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x

class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self,inp,oup,expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SED(nn.Module):
    def __init__(self,dim,out_dim,norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1, bias=False)
        self.norm2 = norm_layer(out_dim)
        self.norm1 = norm_layer(dim)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = x + self.conv(x)
        x = self.reduction(x)
        x = self.norm2(x)
        return x

class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class ConfuSion(nn.Module):
    """ Confusion model for cnn and vig
    """
    def __init__(self,gnnChannel,cnnChannel):
        super().__init__()
        self.gnnConv = nn.Conv2d(gnnChannel+cnnChannel, gnnChannel, 1, 1, 0, bias=False)
        self.cnnConv = nn.Conv2d(gnnChannel+cnnChannel, cnnChannel, 1, 1, 0, bias=False)
    def forward(self, x_gnn, x_cnn):
        x = torch.cat((x_gnn,x_cnn),dim=1)
        x_g = self.gnnConv(x)
        x_c = self.cnnConv(x)
        return x_g,x_c
class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        
        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)] 
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)] 
        max_dilation = 49 // max(num_knn) #5
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        HW = 224 // 4 * 224 // 4
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(SED(channels[i-1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                    relative_pos=True),
                          FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                         )]
                idx += 1
        self.backbone = Seq(*self.backbone)
        self.prediction = Seq(nn.Conv2d(channels[-1], emb_dims, 1, bias=True),
                              nn.BatchNorm2d(emb_dims),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(emb_dims, opt.num_classes, 1, bias=True))

        self.transition0=ConfuSion(channels[0],256)
        self.transition1=ConfuSion(channels[1],512)
        self.transition2=ConfuSion(channels[2],512)
        self.transition3=ConfuSion(channels[3],512)
        self.w = nn.Parameter(torch.ones(2))
        self.model_init()
        self.cnn_backbone = pretrained_model(net_type=opt.net_type,depth=opt.depth,num_classes=opt.num_classes)
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x_gnn = self.stem(inputs) + self.pos_embed 
        x_cnn = self.cnn_backbone.stem(inputs) 
        for i in range(len(self.backbone)):
            if(i%4==0):
                x_gnn = self.backbone[i](x_gnn)
                i=i//4
                attr_name = 'conv' + str(i)
                attr_name_gnn = 'transition' + str(i)
                if hasattr(self, attr_name_gnn):
                    downsample_attr_gnn = getattr(self, attr_name_gnn)
                    x_gnn,x_cnn = downsample_attr_gnn(x_gnn,x_cnn)
                if hasattr(self.cnn_backbone, attr_name):
                    downsample_attr = getattr(self.cnn_backbone, attr_name)
                    x_cnn = downsample_attr(x_cnn)
            else:
                x_gnn = self.backbone[i](x_gnn)
        x_cnn = x_cnn.view(x_cnn.size(0), -1)
        x_cnn = self.cnn_backbone.fc(x_cnn)
        x_gnn = F.adaptive_avg_pool2d(x_gnn, 1)
        x_gnn = self.prediction(x_gnn).squeeze(-1).squeeze(-1)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        return w1*x_gnn,w2*x_cnn


@register_model
def Cpvig_ti_224_gelu(args):
    model = DeepGCN(args)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,6,2] # number of basic blocks in the backbone
            self.channels = [80, 160, 400, 640] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_m_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,16,2] # number of basic blocks in the backbone
            self.channels = [96, 192, 384, 768] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']
    return model


@register_model
def pvig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.k = 9 # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,18,2] # number of basic blocks in the backbone
            self.channels = [128, 256, 512, 1024] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['vig_b_224_gelu']
    return model
