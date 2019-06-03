from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import os
import cv2
import lmdb
import sys

from caffe2.python import workspace, model_helper, core, brew, net_drawer
from caffe2.proto import caffe2_pb2



# define resnet50-type model structure
def meta_conv(
    model,
    inputs,
    dim_in,
    dim_out,
    kernel,
    pad,
    stride,
    no_bias=False,
    is_test=False,
    has_relu=False,
    module_seq=None,
    sub_seq=None,
    branch_seq=None,
    conv_seq=None
):
    '''
    add basic conv module of resnet 50
    '''
    # convolution layer
    conv = brew.conv(
        model,
        inputs,
        '{}_{}_branch{}{}'.format(module_seq, sub_seq, branch_seq, conv_seq),
        dim_in=dim_in,
        dim_out=dim_out,
        kernel=kernel,
        stride=stride,
        pad=pad,
        no_bias=no_bias,
    )

    # spaial batchnormalization layer
    bn = brew.spatial_bn(
        model,
        conv,
        '{}_{}_branch{}{}_bn'.format(module_seq, sub_seq, branch_seq, conv_seq),
        # conv,    # in-place
        dim_in = dim_out,
        epsilon = 1e-05,
        is_test = is_test,
    )

    # ReLU layer
    if has_relu:
        relu = brew.relu(model, bn, bn)
        return relu
    else:
        return bn


def input_block(
    model,
    inputs,
    dim_in,
    dim_out,
    kernel,
    pad,
    stride,
    no_bias=False,
    is_test=False,
):
    '''
    add input conv module (separated out due to the name of predict.pbtxt)
    '''
    # convolution layer
    conv = brew.conv(
        model,
        inputs,
        'conv1',
        dim_in=dim_in,
        dim_out=dim_out,
        kernel=kernel,
        pad=pad,
        stride=stride,
        no_bias=no_bias
    )

    # spaial batchnormalization layer
    bn = brew.spatial_bn(
        model,
        conv,
        'res_conv1_bn',
        # conv,    # in-place
        dim_in = dim_out,
        epsilon = 1e-05,
        is_test = is_test,
    )

    # ReLU layer
    relu = brew.relu(
        model,
        bn,
        bn # in-place
    )

    # max pool
    pool = brew.max_pool(
        model,
        relu,
        'pool1',
        kernel=3,
        stride=2,
        pad=1,
    )

    return pool


def res_block_1(
    model,
    inputs,
    dim_in,
    dim_mid,
    dim_out,
    kernel,  # for first conv
    pad,
    stride,
    no_bias=False,
    is_test=False,
    module_seq=None,
    sub_seq=None,
):
    # branch1 (left)
    branch1_conv = meta_conv(
        model,
        inputs,
        dim_in,
        dim_out,
        kernel,
        pad,
        stride,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=False,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='1',
        conv_seq='',
    )

    # branch2 (right)
    branch2_conv1 = meta_conv(
        model,
        inputs,
        dim_in,
        dim_mid,
        kernel,
        pad,
        stride,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=True,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='a',
    )

    branch2_conv2 = meta_conv(
        model,
        branch2_conv1,
        dim_mid,
        dim_mid,
        kernel=3,
        pad=1,
        stride=1,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=True,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='b',
    )

    branch2_conv3 = meta_conv(
        model,
        branch2_conv2,
        dim_mid,
        dim_out,
        kernel=1,
        pad=0,
        stride=1,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=False,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='c',
    )

    # sum
    branch_sum = brew.sum(
        model,
        [branch2_conv3, branch1_conv],
        branch2_conv3
    )
    branch_relu = brew.relu(
        model,
        branch_sum,
        branch_sum
    )
    return branch_relu


def res_block_2(
    model,
    inputs,
    dim_in,
    dim_mid,
    dim_out,
    no_bias=False,
    is_test=False,
    module_seq=None,
    sub_seq=None,
):
    # branch2 (right)
    branch2_conv1 = meta_conv(
        model,
        inputs,
        dim_in,
        dim_mid,
        kernel=1,
        pad=0,
        stride=1,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=True,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='a',
    )

    branch2_conv2 = meta_conv(
        model,
        branch2_conv1,
        dim_mid,
        dim_mid,
        kernel=3,
        pad=1,
        stride=1,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=True,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='b',
    )

    branch2_conv3 = meta_conv(
        model,
        branch2_conv2,
        dim_mid,
        dim_out,
        kernel=1,
        pad=0,
        stride=1,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=False,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='c',
    )

    # sum
    branch_sum = brew.sum(
        model,
        [branch2_conv3, inputs],
        branch2_conv3
    )
    branch_relu = brew.relu(
        model,
        branch_sum,
        branch_sum
    )
    return branch_relu


def res_module(
    model,
    inputs,
    dim_in,
    dim_mid,
    dim_out,
    kernel,  # for first conv
    pad,
    stride,
    no_bias=False,
    is_test=False,
    module_seq=None,
    sub_num=2,
):
    # first block - res_block_1
    block_0 = res_block_1(
        model,
        inputs,
        dim_in=dim_in,
        dim_mid=dim_mid,
        dim_out=dim_out,
        kernel=kernel,
        pad=pad,
        stride=stride,
        no_bias=no_bias,
        is_test=is_test,
        module_seq=module_seq,
        sub_seq='0',
    )

    # following blocks - res_block_2
    blocks = [None for i in range(sub_num)]
    blocks[0] = block_0
    for i in range(1, sub_num):
        blocks[i] = res_block_2(
            model,
            blocks[i - 1],
            dim_in=dim_out,
            dim_mid=dim_mid,
            dim_out=dim_out,
            no_bias=no_bias,
            is_test=is_test,
            module_seq=module_seq,
            sub_seq=str(i)
        )

    return blocks[-1]


def add_resnet50_finetune(model, data, num_class=2, is_test=False):
    '''
    construct resnet50 net for finetune
    default remove last fc
    '''
    pool_1 = input_block(model, data, 3, 64, 7, 3, 2, no_bias=False, is_test=is_test)
    module_2 = res_module(model, pool_1, 64, 64, 256, 1, 0, 1,
                          no_bias=True, is_test=is_test, module_seq='res2', sub_num=3)
    module_3 = res_module(model, module_2, 256, 128, 512, 1, 0, 2,
                          no_bias=True, is_test=is_test, module_seq='res3', sub_num=4)
    module_4 = res_module(model, module_3, 512, 256, 1024, 1, 0, 2,
                          no_bias=True, is_test=is_test, module_seq='res4', sub_num=6)
    module_5 = res_module(model, module_4, 1024, 512, 2048, 1, 0, 2,
                          no_bias=True, is_test=is_test, module_seq='res5', sub_num=3)
    pool_5 = brew.average_pool(model, module_5, 'pool5', kernel=7, stride=1)
    # finetune part
    finetune_fc = brew.fc(model, pool_5, 'finetune_fc', dim_in=2048, dim_out=num_class)
    softmax = brew.softmax(model, finetune_fc, 'softmax')

    return softmax


def add_resnet50(model, data, is_test=False):
    '''
    construct resnet50 net
    '''
    pool_1 = input_block(model, data, 3, 64, 7, 3, 2, no_bias=False, is_test=is_test)
    module_2 = res_module(model, pool_1, 64, 64, 256, 1, 0, 1,
                          no_bias=True, is_test=is_test, module_seq='res2', sub_num=3)
    module_3 = res_module(model, module_2, 256, 128, 512, 1, 0, 2,
                          no_bias=True, is_test=is_test, module_seq='res3', sub_num=4)
    module_4 = res_module(model, module_3, 512, 256, 1024, 1, 0, 2,
                          no_bias=True, is_test=is_test, module_seq='res4', sub_num=6)
    module_5 = res_module(model, module_4, 1024, 512, 2048, 1, 0, 2,
                          no_bias=True, is_test=is_test, module_seq='res5', sub_num=3)
    pool_5 = brew.average_pool(model, module_5, 'pool5', kernel=7, stride=1)
    pred = brew.fc(model, pool_5, 'pred', dim_in=2048, dim_out=1000)

    return pred


def add_resnet50_core(model, data, is_test=False):
    ''' construct resnet50 core for finetune, default remove last fc
    Args:
        model: model_helper instance
        data: 'data' BlobRef
        is_test: bool denotes training or testing model
    Returns:
        core_output: BlobRef of the output of the core net
        dim_out: a int32 of the dim of the core_output
    '''
    pool_1 = input_block(model, data, 3, 64, 7, 3, 2, no_bias=False, is_test=is_test)
    module_2 = res_module(model, pool_1, 64, 64, 256, 1, 0, 1,
                          no_bias=True, is_test=is_test, module_seq='res2', sub_num=3)
    module_3 = res_module(model, module_2, 256, 128, 512, 1, 0, 2,
                          no_bias=True, is_test=is_test, module_seq='res3', sub_num=4)
    module_4 = res_module(model, module_3, 512, 256, 1024, 1, 0, 2,
                          no_bias=True, is_test=is_test, module_seq='res4', sub_num=6)
    module_5 = res_module(model, module_4, 1024, 512, 2048, 1, 0, 2,
                          no_bias=True, is_test=is_test, module_seq='res5', sub_num=3)

    # return last res-block conv
    return module_5


if __name__ == '__main__':
    model = model_helper.ModelHelper('structure_test')
    data = model.net.ConstantFill([], ['data'], shape=[1, 3, 224, 224], value=1.0)
    add_resnet50_finetune(model, data)

    init_net_pb = '/home/zhibin/qzhong/caffe2/caffe2_model_zoo/resnet50/resnet50_init_net.pb'
    init_net_proto = caffe2_pb2.NetDef()
    with open(init_net_pb, 'rb') as f:
        init_net_proto.ParseFromString(f.read())

    model.param_init_net.AppendNet(core.Net(init_net_proto))
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    workspace.RunNet(model.net)



