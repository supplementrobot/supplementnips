
import torch.nn as nn
import torch
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from network.graph_attention_layer import GraphAttentionLayer
from torchdiffeq import odeint, odeint_adjoint
from network.swinblock import SwinBlock




# -- CLIP residualattentionblock
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, batch_first=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x














# -- basickblock bottleneck
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # expansion: int = 4
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class ODEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True, tol=1e-3) -> None:
        super().__init__()
        self.tol = tol
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first


        class ODEFunc(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seq = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first)
            def forward(self, t, x):
                output, _ = self.seq(x,x,x)
                return output
            
        self.ode_func = ODEFunc()



    def forward(self, x):
        output = odeint_adjoint(self.ode_func, y0=x, t=torch.tensor([0.,1.]).to(device=x.device), method='dopri5',
                                atol=self.tol, rtol=self.tol)[-1]
        return output
    
    





class QKVAttention(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads


    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, -1) # [batch, seq_len, num_heads, head_dim]
        k = k.view(k.shape[0], k.shape[1], self.num_heads, -1)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, -1)
        att = q @ k.permute(0,1,3,2) # [batch, seq_len, num_heads, num_heads]

        output = att @ v # [batch, seq_len, num_heads, head_dim]
        output = output.view(output.shape[0], output.shape[1], -1) # [batch, seq_len, embed_dim]

        return output





class QKVAttentionM(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads 
        self.M = nn.Parameter(torch.randn(self.head_dim, self.head_dim), requires_grad=True)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, -1) # [batch, seq_len, num_heads, head_dim]
        k = k.view(k.shape[0], k.shape[1], self.num_heads, -1)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, -1)

        att = q @ self.M @ k.permute(0,1,3,2) # [batch, seq_len, num_heads, num_heads]

        output = att @ v # [batch, seq_len, num_heads, head_dim]
        output = output.view(output.shape[0], output.shape[1], -1) # [batch, seq_len, embed_dim]

        return output





class QKVAttentionG1(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.g = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        head_dim = embed_dim // num_heads 

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        g = self.g(x)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, -1) # [batch, seq_len, num_heads, head_dim]
        k = k.view(k.shape[0], k.shape[1], self.num_heads, -1)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, -1)
        g = g.view(g.shape[0], g.shape[1], self.num_heads, -1)

        att = (q * g) @ k.permute(0,1,3,2)

        output = att @ v # [batch, seq_len, num_heads, head_dim]
        output = output.view(output.shape[0], output.shape[1], -1) # [batch, seq_len, embed_dim]

        return output






class QKVAttentionG2(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads 
        self.head_dim = head_dim

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.g = nn.Linear(embed_dim, embed_dim*head_dim)


    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        g = self.g(x)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, -1) # [batch, seq_len, num_heads, head_dim]
        k = k.view(k.shape[0], k.shape[1], self.num_heads, -1)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, -1)
        g = g.view(g.shape[0], g.shape[1], self.num_heads, -1) # [batch, seq_len, num_heads, head_dim*head_dim]

        g = g.view(g.shape[0], g.shape[1], self.num_heads, self.head_dim, self.head_dim) # [batch, seq_len, num_heads, head_dim, head_dim]


        _q = q.unsqueeze(-2) # [batch, seq_len, num_heads, 1, head_dim]

        _att = _q @ g 
        _att = _att.squeeze(-2) # [batch, seq_len, num_heads, head_dim]
        _att = _att @ k.permute(0,1,3,2) # [batch, seq_len, num_heads, num_heads]

        output = _att @ v # [batch, seq_len, num_heads, head_dim]
        output = output.view(output.shape[0], output.shape[1], -1) # [batch, seq_len, embed_dim]

        return output
        








# -- gatt attn are from other places

class GattBlock(nn.Module):
    def __init__(self, block_type, num_heads, in_features, window_size=None, shift_size=None) -> None:
        super().__init__()


        assert block_type in ['basicblock','bottleneck','gatt','gattm','attn','resattn','swinblock',
                              'qkv','qkvm','qkvg1','qkvg2']


        if block_type == 'basicblock':
            block = BasicBlock(inplanes=in_features, planes=in_features)

        elif block_type == 'bottleneck':
            block = Bottleneck(inplanes=in_features, planes=in_features)

        elif block_type == 'gatt':
            block = GraphAttentionLayer(in_features=in_features, hidden_features=in_features//num_heads, n_heads=num_heads, learn_m=False)

        elif block_type == 'gattm':
            block = GraphAttentionLayer(in_features=in_features, hidden_features=in_features//num_heads, n_heads=num_heads, learn_m=True)

        elif block_type == 'attn':
            block = nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads, batch_first=True)



        elif block_type == 'qkv':
            block = QKVAttention(embed_dim=in_features, num_heads=num_heads)

        elif block_type == 'qkvm':
            block = QKVAttentionM(embed_dim=in_features, num_heads=num_heads)

        elif block_type == 'qkvg1':
            block = QKVAttentionG1(embed_dim=in_features, num_heads=num_heads)

        elif block_type == 'qkvg2':
            block = QKVAttentionG2(embed_dim=in_features, num_heads=num_heads)




        elif block_type == 'resattn':
            block = ResidualAttentionBlock(d_model=in_features, n_head=num_heads, batch_first=True)
            
        elif block_type == 'swinblock':
            block = SwinBlock(dim=in_features, num_heads=num_heads, window_size=window_size, shift_size=shift_size,
                              mlp_ratio=4.0, dropout=0, attention_dropout=0, stochastic_depth_prob=0)




        # block = ODEMultiheadAttention(embed_dim=in_features, num_heads=num_heads, batch_first=True)


        self.block_type = block_type
        self.block = block



    def forward(self, x, h=None, w=None):
        # x:[b,hw,c]
        assert len(x.shape) == 3
        b,hw,c = x.shape
        
        

        if self.block_type in ['basicblock','bottleneck']:
            assert h is not None
            x = x.permute(0,2,1).view(b,c,h,w)
            output = self.block(x)
            output = output.view(b,c,hw).permute(0,2,1)

     
        elif self.block_type == 'gatt':
            output = self.block(x)


        elif self.block_type == 'gattm':
            output = self.block(x)


        elif self.block_type == 'attn':
            output = self.block(x,x,x)
            output = output[0]




        elif self.block_type == 'qkv':
            output = self.block(x)

        elif self.block_type == 'qkvm':
            output = self.block(x)

        elif self.block_type == 'qkvg1':
            output = self.block(x)

        elif self.block_type == 'qkvg2':
            output = self.block(x)




        elif self.block_type == 'resattn':
            output = self.block(x)


        elif self.block_type == 'swinblock':
            x = x.view(b,h,w,c)
            output = self.block(x)
            output = output.view(b,hw,c)




        return output




if __name__ == '__main__':

    model = GattBlock(block_type='attn', num_heads=8, in_features=128)


    x = torch.rand([4, 300, 128])

    output = model(x)

    a=1