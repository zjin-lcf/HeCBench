#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import intel_extension_for_pytorch
from dcn_v2 import DCN

def example_dconv():
    torch.manual_seed(0)
    input = torch.randn(100, 64, 128, 128).xpu()
    dcn = DCN(64, 64, kernel_size=(3, 3), stride=1,
              padding=1, deformable_groups=2).xpu()
    # print(dcn.weight.shape, input.shape)
    output = dcn(input)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    #print("Output shape: ", output.shape)
    print("Mean error: ", error)


if __name__ == '__main__':
    example_dconv()
