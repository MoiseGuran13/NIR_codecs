import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
#import model.MP_IFneuron
# from torch.cuda.amp import autocast as autocast
from torch.nn.utils.rnn import pad_sequence
from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate, base, layer
import math
from math import ceil, floor
from torch.nn import ZeroPad2d
try:
    import cupy
    from . import neuron_kernel, cu_kernel_opt
except ImportError:
    neuron_kernel = None


def optimal_crop_size(max_size, max_subsample_factor, safety_margin=0):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    crop_size += safety_margin * pow(2, max_subsample_factor)
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders, safety_margin=0):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders, safety_margin)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders, safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ZeroPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)


def normalize_image(image, percentile_lower=1, percentile_upper=99):
    mini, maxi = np.percentile(image, (percentile_lower, percentile_upper))
    if mini == maxi:
        return 0 * image + 0.5  # gray image
    return np.clip((image - mini) / (maxi - mini + 1e-5), 0, 1)


class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):

        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor):

        self.neuronal_charge(x)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike

class BaseNode_adaspike(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):

        raise NotImplementedError

    def neuronal_fire(self):

        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):

        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor, s: torch.Tensor):

        self.neuronal_charge(x, s)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike

class MpNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        
        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)


        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor, last_mem: torch.Tensor):
        if last_mem is None:
            self.neuronal_charge(x)
        else:
            self.register_memory('v', last_mem)
            self.neuronal_charge(x)
        return self.v

class Ada_MpNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        
        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)


        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor, last_mem: torch.Tensor, w: torch.Tensor):
        if last_mem is None:
            self.neuronal_charge(x, w)
        else:
            self.register_memory('v', last_mem)
            self.neuronal_charge(x, w)

        return self.v

class Ada_MpNode_adaspike(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        
        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)


        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor, last_mem: torch.Tensor, w: torch.Tensor, s: torch.Tensor):
        if last_mem is None:
            self.neuronal_charge(x, w, s)
        else:
            self.register_memory('v', last_mem)
            self.neuronal_charge(x, w, s)
        return self.v

class Multi_Node(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor, last_mem: torch.Tensor):
        if last_mem is None:
            self.neuronal_charge(x)
            self.neuronal_fire()
            self.neuronal_reset()
        else:
            self.register_memory('v', last_mem)
            self.neuronal_charge(x)
            self.neuronal_fire()
            self.neuronal_reset()

        return self.spike, self.v

class MpLIFNode(MpNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'
 
    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

class Mp_AdaLIFNode(Ada_MpNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(tau, float) and tau > 1.
        self.tau = tau
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'
 
    def neuronal_charge(self, x: torch.Tensor, w: torch.Tensor):
        tau = w.sigmoid()
        if self.v_reset is None:
            self.v = self.v + (x - self.v) * tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) * tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * tau

class Mp_AdaLIFNode_adaspike(Ada_MpNode_adaspike):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(tau, float) and tau > 1.
        self.tau = tau
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'
 
    def neuronal_charge(self, x: torch.Tensor, w: torch.Tensor, s: torch.Tensor):
        tau = w.sigmoid()
        if self.v_reset is None:
            self.v = self.v + s * (x - self.v) * tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) * tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * tau

class MpIFNode(MpNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

class Mp_ParametricLIFNode(MpNode):
    def __init__(self, init_tau: float = 2.0, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def extra_repr(self):
        with torch.no_grad():
            tau = self.w.sigmoid()  #.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) * self.w.sigmoid()
        else:
            if self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()

class Mp_ParametricLIFNode_modify(MpNode):
    def __init__(self, size_h, size_w, init_tau: float = 2.0, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.ones(size=[size_w, size_h])* init_w)
        #self.w = self.w * init_w   # test

    def extra_repr(self):
        with torch.no_grad():
            tau = self.w.sigmoid()  #.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) * self.w.sigmoid()
        else:
            if self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()

class IFNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

class LIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

class LIFNode_adaspike(BaseNode_adaspike):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor, s: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + s*(x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau


class ParametricLIFNode(BaseNode):
    def __init__(self, init_tau: float = 2.0, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def extra_repr(self):
        with torch.no_grad():
            tau = self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) * self.w.sigmoid()
        else:
            if self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()


class BasicModel(nn.Module):
    '''
    Basic model class that can be saved and loaded
        with specified names.
    '''

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print('save model to \"{}\"'.format(path))

    def load(self, path: str):
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state)
            print('load pre-trained model \"{}\"'.format(path))
        else:
            print('init model')
        return self
    
    def to(self, device: torch.device):
        self.device = device
        super().to(device)
        return self

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm=None, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(ConvLayer, self).__init__()

        bias = False
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        if activation_type == 'lif':
            self.activation = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'if':
            self.activation = IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'plif':
            self.activation = ParametricLIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv2d(x)
        out = self.norm_layer(out)
        out = self.activation(out)
        return out

class Spike_recurrentConvLayer_nolstm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm=None, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(Spike_recurrentConvLayer_nolstm, self).__init__()

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, norm, tau, v_threshold, v_reset, activation_type)
        #self.recurrent_block = rnn.SpikingConvLSTMCell(input_size=out_channels, hidden_size=out_channels)

    def forward(self, x):
        x = self.conv(x)
        # state = self.recurrent_block(x, prev_state)
        # x = state[0]
        return x

class Spike_skip_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm=None, tau=2, v_threshold=1.0, v_reset=None, activation_type='lif'):
        super(Spike_skip_layer, self).__init__()
        self.conv = ConvLayer_ada_simmp(in_channels, out_channels, kernel_size, stride, padding, norm, tau, v_threshold, v_reset, activation_type)

    def forward(self, x, last_mem):
        x = self.conv(x, last_mem)
        return x

class ConvLayer_ada_simmp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm=None, tau=2, v_threshold=1.0, v_reset=None, activation_type='lif'):
        super(ConvLayer_ada_simmp, self).__init__()

        bias = False
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        #self.activation = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.norm_layer = nn.BatchNorm2d(out_channels)
        self.conv2d_pool = nn.Conv2d(out_channels, 1, kernel_size, stride, padding, bias=bias)
        #self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.AdaptiveMaxPool2d(1)
        if activation_type == 'plif':
            self.activation = Mp_ParametricLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()) 
        elif activation_type == 'lif':
            self.activation = MpLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()) 
        elif activation_type == 'if':
            self.activation = MpIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'ada_lif':
            self.activation = Mp_AdaLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 4, 2*2)
        self.sigmoid = nn.Sigmoid()
        #A
        self.get_theta = get_theta_simmp(channels1 = in_channels, channels2 = out_channels, type='global', type1 = 'mix')
        #B
        #self.get_theta = get_theta(channels = in_channels, type='channel', type1 = 'fr')

    def forward(self, x, last_mem):
        out = self.conv2d(x)
        out = self.norm_layer(out)

        w = self.get_theta(x, out)
        out = self.activation(out, last_mem, w.unsqueeze(-1).unsqueeze(-1))
        return out

class get_theta_simmp(nn.Module):
    def __init__(self, channels1, channels2, reduction=4, type='global', type1 = 'max'):
        super(get_theta_simmp, self).__init__()
        self.channels = channels1
        self.fc1 = nn.Linear(channels1, channels1 // reduction)
        self.relu = nn.ReLU(inplace=True)
        if type == 'global':
            self.fc2 = nn.Linear(channels2// reduction, 1)
        elif type == 'channel':
            self.fc2 = nn.Linear(channels2 // reduction, channels1)
        self.sigmoid = nn.Sigmoid() 
        if type1 == 'max':    
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif type1 == 'fr':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif type1 == 'mix':
            self.pool = nn.AdaptiveMaxPool2d(1)
            self.pool1 = nn.AdaptiveAvgPool2d(1)
            self.fc3 = nn.Linear(channels1+channels2, channels2 // reduction)

    def forward(self, x, x1):
        if x1 is None:
            theta = self.pool(x)
            theta = self.fc1(theta.squeeze(-1).squeeze(-1))
            theta = self.relu(theta)
            theta = self.fc2(theta)
        else:
            theta1 = self.pool(x1)
            theta2 = self.pool1(x)
            theta = torch.cat([theta1,theta2],1)
            theta = self.fc3(theta.squeeze(-1).squeeze(-1))
            theta = self.relu(theta)
            theta = self.fc2(theta)            
        return theta

class MP_upsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=False):
        super(MP_upsample_layer, self).__init__()

        bias = False
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        #self.activation = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        out = self.norm_layer(out)
        return out

class Spiking_residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, tau=2, v_threshold=1.0, v_reset=None):
        super(Spiking_residualBlock, self).__init__()
        bias = False
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.lif = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        #self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        
        out = self.bn1(out)
        out = self.lif(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.lif(out)
        return out

class Spike_upsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm=None, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(Spike_upsample_layer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if activation_type == 'lif':
            self.activation = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'if':
            self.activation = IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'plif':
            self.activation = ParametricLIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan())        
        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        out = self.norm_layer(out)
        out = self.activation(out)

        return out

class TemporalFlatLayer_ada_simmp_concat(nn.Module):
    def __init__(self, tau=2.0, v_reset=None, activation_type='plif'):
        super(TemporalFlatLayer_ada_simmp_concat, self).__init__()

        self.conv2d = nn.Conv2d(64, 32, 1, bias=False)
        self.norm_layer = nn.BatchNorm2d(32)
        self.conv2d_pool = nn.Conv2d(32, 1, 1, bias=False)
        #self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.AdaptiveMaxPool2d(1)
        if activation_type == 'plif':
            self.activation = Mp_ParametricLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()) 
        elif activation_type == 'lif':
            self.activation = MpLIFNode(v_threshold=float('Inf'), tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'if':
            self.activation = MpIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()) 
        elif activation_type == 'ada_lif':
            self.activation = Mp_AdaLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())            
        self.get_theta = get_theta_simmp(channels1 = 64, channels2 = 32, type='global', type1 = 'mix')

    def forward(self, x, last_mem):
        out = self.conv2d(x)
        out = self.norm_layer(out)
        w = self.get_theta(x, out)
        out = self.activation(out, last_mem, w.unsqueeze(-1).unsqueeze(-1))
        return out

class TemporalFlatLayer_concat(nn.Module):
    def __init__(self, tau=2.0, v_reset=None):
        super(TemporalFlatLayer_concat, self).__init__()

        self.conv2d = nn.Conv2d(64, 1, 1, bias=False)
        self.norm_layer = nn.BatchNorm2d(1)
        self.activation = MpLIFNode(v_threshold=float('Inf'), tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        #self.activation = MpIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())
        #self.activation = Mp_ParametricLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()) 
                           
    def forward(self, x, last_mem):
        out = self.conv2d(x)
        out = self.norm_layer(out)
        out = self.activation(out, last_mem)
        return out

class PAEVSNN_LIF_AMPLIF_final(BasicModel):

    def __init__(self, kwargs = {}):
        super().__init__()
        activation_type =  kwargs['activation_type']
        mp_activation_type = kwargs['mp_activation_type']
        spike_connection = kwargs['spike_connection']
        v_threshold = kwargs['v_threshold']
        v_reset = kwargs['v_reset']
        tau = kwargs['tau']

        #header
        mp_activation_type = 'ada_lif'
        activation_type = 'lif'

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

        self.down1 = Spike_recurrentConvLayer_nolstm(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.down2 = Spike_recurrentConvLayer_nolstm(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.down3 = Spike_recurrentConvLayer_nolstm(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)

        self.skip0 = Spike_skip_layer(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)
        self.skip1 = Spike_skip_layer(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)
        self.skip2 = Spike_skip_layer(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)
        self.skip3 = Spike_skip_layer(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)

        self.up1mp = Spike_skip_layer(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)
        self.up2mp = Spike_skip_layer(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)
        self.up3mp = Spike_skip_layer(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)

        self.aggregation1 = MP_upsample_layer(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, output_padding=0)
        self.aggregation2 = MP_upsample_layer(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, output_padding=0)
        self.aggregation3 = MP_upsample_layer(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, output_padding=0)

        self.residualBlock = nn.Sequential(
            Spiking_residualBlock(256, 256, stride=1, tau=tau, v_threshold=v_threshold, v_reset=v_reset),
        )

        self.up1 = Spike_upsample_layer(in_channels=512, out_channels=128, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.up2 = Spike_upsample_layer(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.up3 = Spike_upsample_layer(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)

        self.temporalflat = TemporalFlatLayer_ada_simmp_concat(tau = tau, v_reset=None, activation_type = mp_activation_type)

        self.final = nn.Sequential(
            nn.Conv2d(32, 1, 1, bias=False), 
        )
            
    def forward(self, on_img, prev_mem_states):
        if prev_mem_states is None:
            prev_mem_states = [None] * 8

        for i in range(on_img.size(1)):
            mem_states = []

            x = on_img[:,i,:,:].unsqueeze(dim=1)
            x_in = self.static_conv(x)  #head

            x1 = self.down1(x_in) #encoder
            x2 = self.down2(x1)
            x3 = self.down3(x2)

            s0 = self.skip0(x_in, prev_mem_states[0])
            mem_states.append(s0)
            s1 = self.skip1(x1, prev_mem_states[1])
            mem_states.append(s1)
            s2 = self.skip2(x2, prev_mem_states[2])
            mem_states.append(s2)
            s3 = self.skip3(x3, prev_mem_states[3])
            mem_states.append(s3)

            r1 = self.residualBlock(x3)

            u1 = self.up1(torch.cat([r1, x3],1)) #decoder
            u2 = self.up2(torch.cat([u1, x2],1))
            u3 = self.up3(torch.cat([u2, x1],1))

            up1mp = self.up1mp(r1, prev_mem_states[4])
            mem_states.append(up1mp)
            Mp1 = s3 + up1mp
            up2mp = self.up2mp(u1, prev_mem_states[5])
            mem_states.append(up2mp)
            Mp2 = s2 + up2mp
            up3mp = self.up3mp(u2, prev_mem_states[6])
            mem_states.append(up3mp)
            Mp3 = s1 + up3mp

            a1 = self.aggregation1(Mp1)
            a2 = self.aggregation2(a1+Mp2)
            a3 = self.aggregation3(a2+Mp3)

            membrane_potential = self.temporalflat(torch.cat([u3,x_in],1), prev_mem_states[7])
            mem_states.append(membrane_potential)
            
            membrane_potential = self.final(membrane_potential+a3+s0)

        return membrane_potential, mem_states

class EVSNN_LIF_final(BasicModel):

    def __init__(self, kwargs = {}):
        super().__init__()
        activation_type =  kwargs['activation_type']
        mp_activation_type = kwargs['mp_activation_type']
        spike_connection = kwargs['spike_connection']
        v_threshold = kwargs['v_threshold']
        v_reset = kwargs['v_reset']
        tau = kwargs['tau']
        self.num_encoders = kwargs['num_encoders']
        self.states = [None] * self.num_encoders

        #header
        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.down1 = Spike_recurrentConvLayer_nolstm(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.down2 = Spike_recurrentConvLayer_nolstm(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.down3 = Spike_recurrentConvLayer_nolstm(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)

        self.residualBlock = nn.Sequential(
            Spiking_residualBlock(256, 256, stride=1, tau=tau, v_threshold=v_threshold, v_reset=v_reset),
        )

        self.up1 = Spike_upsample_layer(in_channels=512, out_channels=128, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.up2 = Spike_upsample_layer(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.up3 = Spike_upsample_layer(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)

        self.temporalflat = TemporalFlatLayer_concat(tau = tau, v_reset=None)

            
    def reset_states(self):
        self.states = [None] * self.num_encoders


    def forward(self, on_img):
        for i in range(on_img.size(1)):
            x = on_img[:,i,:,:].unsqueeze(dim=1)

            x_in = self.static_conv(x) #.unsqueeze(dim=1)

            x1 = self.down1(x_in)
            x2 = self.down2(x1)
            x3 = self.down3(x2)

            r1 = self.residualBlock(x3)

            u1 = self.up1(torch.cat([r1,x3],1))
            u2 = self.up2(torch.cat([u1, x2],1))
            u3 = self.up3(torch.cat([u2, x1],1))
            
            membrane_potential = self.temporalflat(torch.cat([u3,x_in],1), self.states)
            self.states = membrane_potential

        return membrane_potential