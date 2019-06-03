from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import scipy
from scipy import sparse
import os
import time
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from caffe2.python import workspace, model_helper, core, brew, utils, optimizer
from caffe2.python.predictor import mobile_exporter
from caffe2.proto import caffe2_pb2

from add_resnet50_model import add_resnet50_core



##############################################################################
# model maintaining utils
##############################################################################
def load_model(model, init_net_pb, predict_net_pb):
    ''' load init and predict net from .pb file for model validation/testing
        model: current model
        init_net: the .pb file of the init_net
        predict_net: the .pb file of the predict_net
    '''
    # Make sure both nets exists
    if (not os.path.exists(init_net_pb)) or (not os.path.exists(predict_net_pb)):
            print("ERROR: input net.pb not found!")

    # Append net
    init_net_proto = caffe2_pb2.NetDef()
    with open(init_net_pb, 'r') as f:
        init_net_proto.ParseFromString(f.read())
    model.param_init_net = model.param_init_net.AppendNet(core.Net(init_net_proto))

    predict_net_proto = caffe2_pb2.NetDef()
    with open(predict_net_pb, 'r') as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = model.net.AppendNet(core.Net(predict_net_proto))


def load_init_net(init_net_pb, device_opt):
    ''' load params of pretrained init_net on given device '''
    init_net_proto = caffe2_pb2.NetDef()
    with open(init_net_pb, 'rb') as f:
        init_net_proto.ParseFromString(f.read())
        for op in init_net_proto.op:
            op.device_option.CopyFrom(device_opt)
    workspace.RunNetOnce(core.Net(init_net_proto))


def snapshot_init_net(params, workspace, snapshot_prefix, snapshot_name,
                      postfix, epoch):
    ''' save the model init_net as .pb file periodically '''
    timestamp = time.time()
    timestamp_s = time.strftime('%m%d-%H:%M', time.localtime(timestamp))
    init_net_snapshot = os.path.join(
        snapshot_prefix,
        '{}_init_net_{}_epoch-{}_{}.pb'.format(
            snapshot_name, postfix, epoch, timestamp_s),
    )

    init_net_proto = caffe2_pb2.NetDef()
    for param in params:
        blob = workspace.FetchBlob(param)
        shape = blob.shape
        op = core.CreateOperator(
            'GivenTensorFill',
            [],
            [param],
            arg=[
                utils.MakeArgument('shape', shape),
                utils.MakeArgument('values', blob)
            ]
        )
        init_net_proto.op.extend([op])
    with open(init_net_snapshot, 'wb') as f:
        f.write(init_net_proto.SerializeToString())



##############################################################################
# model construction utils
##############################################################################
def add_input(model, config, is_test=False):
    """
    Add an database input data
    """
    if is_test:
        db_reader = model.CreateDB(
            "val_db_reader",
            db=config['evaluate_data']['data_path'],
            db_type=config['evaluate_data']['data_format'],
        )
        data, label = brew.image_input(
            model,
            db_reader,
            ['data', 'label'],
            batch_size=config['evaluate_data']['input_transform']['batch_size'],
            use_gpu_transform=config['evaluate_data']['input_transform']['use_gpu_transform'],
            scale=config['evaluate_data']['input_transform']['scale'],
            crop=config['evaluate_data']['input_transform']['crop_size'],
            mean_per_channel=config['evaluate_data']['input_transform']['mean_per_channel'],
            std_per_channel=config['evaluate_data']['input_transform']['std_per_channel'],
            mirror=True,
            is_test=True,
        )
    else:
        db_reader = model.CreateDB(
            "train_db_reader",
            db=config['training_data']['data_path'],
            db_type=config['training_data']['data_format'],
        )
        data, label = brew.image_input(
            model,
            db_reader,
            ['data', 'label'],
            batch_size=config['training_data']['input_transform']['batch_size'],
            use_gpu_transform=config['training_data']['input_transform']['use_gpu_transform'],
            scale=config['training_data']['input_transform']['scale'],
            crop=config['training_data']['input_transform']['crop_size'],
            mean_per_channel=config['training_data']['input_transform']['mean_per_channel'],
            std_per_channel=config['training_data']['input_transform']['std_per_channel'],
            mirror=True,
            is_test=False,
        )

    # stop bp
    model.StopGradient('data', 'data')
    model.StopGradient('label', 'label')

    return data, label


def add_osme_branch(model, config, feature, seq):
    ''' add OSME module
    model net architecture:
        feature -> avgpool -> fc[1] -> relu -> fc[2] -> sigmoid ->
        mul ->(avgpool -> fc[3] ->) attention
    '''
    # set weight initialization method
    init_policy = "XavierFill" if seq == 1 else "MSRAFill"
    # GAP
    avg_pool = brew.average_pool(
        model,
        feature,
        'osme_GAP1_{}'.format(seq),
        kernel = config['model_arch']['last_conv_size'],
        stride = 1,
    )

    # fc1 (need different initializer)
    fc1 = brew.fc(
        model,
        avg_pool,
        'osme_fc1_{}'.format(seq),
        dim_in=config['model_arch']['feature_dim'],
        dim_out=config['model_arch']['feature_dim'] // config['model_arch']['r'],
        weight_init=(init_policy, {}),
    )

    # relu
    fc1_relu = brew.relu(model, fc1, fc1)
    # fc2
    fc2 = brew.fc(
        model,
        fc1_relu,
        'osme_fc2_{}'.format(seq),
        dim_in=config['model_arch']['feature_dim'] // config['model_arch']['r'],
        dim_out=config['model_arch']['feature_dim'],
        weight_init=(init_policy, {}),
    )

    # sigmoid
    mask = model.net.Sigmoid(fc2, 'osme_mask_{}'.format(seq))
    # channel-wise mul
    attention = model.net.Mul(
        [feature, mask],
        ['osme_excitation_{}'.format(seq)],
        broadcast=1, axis=0,
    )

    # one more GAP
    attention_gap = brew.average_pool(
        model,
        attention,
        'osme_GAP2_{}'.format(seq),
        kernel = config['model_arch']['last_conv_size'],
        stride = 1,
    )
    # fc3
    att_feature = brew.fc(
        model,
        attention_gap,
        'attention_{}'.format(seq),
        dim_in=config['model_arch']['feature_dim'],
        dim_out=config['model_arch']['attention_dim'],
        weight_init=(init_policy, {}),
    )

    return att_feature


def generate_sketch_matrix(model, config, seq):
    input_dim = config['model_arch']['feature_dim']
    output_dim = config['model_arch']['compact_dim']

    # get rand_h and rand_s
    rand_h = config['consts']['rand_h'][seq - 1]
    rand_s = config['consts']['rand_s'][seq - 1]

    # generate sparse matrix
    index_as_row = np.arange(input_dim).astype(np.int32)
    sparse_matrix = sparse.coo_matrix((rand_s, (index_as_row, rand_h)),
                                      shape=(input_dim, output_dim))

    # transform into dense matrix
    dense_matrix = sparse_matrix.todense()

    # add onto caffe2 network
    sketch_matrix = model.net.GivenTensorFill(
        [],
        ['sketch_matrix_{}'.format(seq)],
        values = np.array(dense_matrix).astype(np.float32),
        shape = [input_dim, output_dim],
    )

    return sketch_matrix


def add_count_sketch(model, config, bottom, seq):
    bottom_transposed = model.net.Transpose(
        [bottom],
        ['bottom_{}_transposed'.format(seq)],
        axes=[0, 2, 3, 1],
    )

    bottom_flat, _ = model.net.Reshape(
        [bottom_transposed],
        ['bottom_{}_flat'.format(seq),
         'bottom_{}_transposed_old_shape'.format(seq)],
        shape=(-1, config['model_arch']['feature_dim']),
    )

    sketch_matrix = generate_sketch_matrix(model, config, seq)
    sketch = model.net.MatMul(
        [bottom_flat, sketch_matrix],
        ['sketch_{}'.format(seq)],
    )

    return sketch


def _add_compact_bilinear_pooling(model, config, bottom1, bottom2, sum_pool=True):
    # one more GAP
    bottom1_gap = brew.average_pool(
        model,
        bottom1,
        'bottom1_gap',
        kernel = config['model_arch']['last_conv_size'],
        stride = 1,
    )
    bottom2_gap = brew.average_pool(
        model,
        bottom2,
        'bottom2_gap',
        kernel = config['model_arch']['last_conv_size'],
        stride = 1,
    )
    # fc3
    att1 = brew.fc(
        model,
        bottom1_gap,
        'att1',
        dim_in=config['model_arch']['feature_dim'],
        dim_out=config['model_arch']['compact_dim'],
    )
    att2 = brew.fc(
        model,
        bottom2_gap,
        'att2',
        dim_in=config['model_arch']['feature_dim'],
        dim_out=config['model_arch']['compact_dim'],
    )
    cbp_feature = model.net.Add(
        [att1, att2],
        ['cbp_feature'],
        broadcast=1, axis=0,
    )

    return cbp_feature


def add_compact_bilinear_pooling(model, config, bottom1, bottom2, sum_pool=True):
    """ compute compact bilinear pooling using count sketch, refereced by
    1. https://github.com/DeepInsight-PCALab/CompactBilinearPooling-Pytorch/
    blob/master/CompactBilinearPooling.py
    2. https://github.com/gdlg/pytorch_compact_bilinear_pooling/blob/master/
    compact_bilinear_pooling/__init__.py

    Args:
        model: caffe2 model
        config: config dict
        bottom1: one way of attention feature
        bottom2: the other way of attention feature
    Return:
        cbp_feature: compact form of bilinear attention feature
    """
    # compute count sketch
    sketch1 = add_count_sketch(model, config, bottom1, 1)
    sketch2 = add_count_sketch(model, config, bottom2, 2)

    # FFT & iFFT transformation
    zero_imag = model.net.ZerosLike(sketch1)
    fft1_real, fft1_imag = model.net.FFT(
        [sketch1, zero_imag],
        ['fft1_real', 'fft1_imag'],
    )
    fft2_real, fft2_imag = model.net.FFT(
        [sketch2, zero_imag],
        ['fft2_real', 'fft2_imag'],
    )

    # FFT result product
    r1r2 = model.net.Mul(
        [fft1_real, fft2_real], ['r1r2'],
        broadcast=1, axis=0,
    )
    i1i2 = model.net.Mul(
        [fft1_imag, fft2_imag], ['i1i2'],
        broadcast=1, axis=0,
    )
    fft_product_real = model.net.Sub(
        [r1r2, i1i2], ['fft_product_real'],
        broadcast=1, axis=0,
    )
    i1r2 = model.net.Mul(
        [fft1_imag, fft2_real], ['i1r2'],
        broadcast=1, axis=0,
    )
    r1i2 = model.net.Mul(
        [fft1_real, fft2_imag], ['r1i2'],
        broadcast=1, axis=0,
    )
    fft_product_imag = model.net.Add(
        [i1r2, r1i2], ['fft_product_imag'],
        broadcast=1, axis=0,
    )

    # IFFT
    cbp_real, cbp_imag = model.net.IFFT(
        [fft_product_real, fft_product_imag],
        ['cbp_real', 'cbp_imag'],
    )
    model.net.ZeroGradient([cbp_imag],[])

    # DEBUG::Print
    # model.net.Print(model.net.Shape([cbp_real],['cbp_real_shape']), [])

    # sum pooling
    cbp_reshaped, _ = model.net.Reshape(
        [cbp_real],
        ['cbp_reshaped', 'cbp_real_old_shape'],
        shape=(
            config['training_data']['input_transform']['batch_size'], # N
            config['model_arch']['last_conv_size'], # H
            config['model_arch']['last_conv_size'], # W
            config['model_arch']['compact_dim'], # C
        ),
    )
    cbp_transposed = model.net.Transpose(
        [cbp_reshaped],
        ['cbp_transposed'],
        axes=[0, 3, 1, 2],
    )
    cbp_tmp = model.net.ReduceBackSum([cbp_transposed], ['cbp_tmp']) # N*d*H
    cbp_feature = model.net.ReduceBackSum([cbp_tmp], ['cbp_feature']) # N*d

    return cbp_feature


def add_normalization(model, cbp_feature):
    # add sign square normalization
    feature_sign = model.net.Sign([cbp_feature], ['cbp_sign'])
    # Since we should not do gradient on operator 'Sign' according to the Doc
    model.StopGradient(feature_sign, feature_sign)

    feature_abs = model.net.Abs([cbp_feature], ['cbp_abs'])
    eps = model.net.ConstantFill([], ['eps'], value=1e-7)
    feature_eps = model.net.Add([feature_abs, eps], ['feature_eps'], broadcast=1)
    feature_sqrt = model.net.Sqrt([feature_eps], ['cbp_sqrt'])

    feature_sign_sqrted = model.net.Mul(
        [feature_sign, feature_sqrt],
        ['cbp_feature_sign_sqrted'],
        broadcast=1,
        axis=0,
    )

    # add L2 normalization
    feature_l2 = model.net.Normalize(
        [feature_sign_sqrted],
        ['cbp_feature_l2'],
    )

    return feature_l2


def _add_model(model, config, data, is_test=False):
    # add back-bone network (resnet-50 with last conv)
    bottom_feature = add_resnet50_core(model, data, is_test=is_test)

    gap_feature = brew.average_pool(
        model,
        bottom_feature,
        'gap_feature',
        kernel = config['model_arch']['last_conv_size'],
        stride=1
    )

    # add prediction for classification
    pred = brew.fc(
        model,
        gap_feature,
        'bi_attention_pred',
        dim_in=config['model_arch']['feature_dim'],
        dim_out=config['model_arch']['num_classes'],
    )

    return pred


def add_model(model, config, data, is_test=False):
    # add back-bone network (resnet-50 with last conv)
    bottom_feature = add_resnet50_core(model, data, is_test=is_test)

    # add osme attention module with different weight initialization
    att1_feature = add_osme_branch(model, config, bottom_feature, 1)
    att2_feature = add_osme_branch(model, config, bottom_feature, 2)
    combined_feature = brew.concat(
        model,
        [att1_feature, att2_feature],
        'combined_feature',
    )

    # add prediction for classification
    pred = brew.fc(
        model,
        combined_feature,
        'bi_attention_pred',
        dim_in=config['model_arch']['feature_dim'],
        dim_out=config['model_arch']['num_classes'],
    )

    return pred


def add_loss(model, config, pred, label):
    # compute softmax loss
    softmax, softmax_loss = model.net.SoftmaxWithLoss(
        [pred, label],
        ['softmax', 'softmax_loss'],
    )

    # compute PC reg
    left, right = model.net.Split(
        [pred],
        ['left', 'right'],
        split=(config['training_data']['input_transform']['batch_size'] // 2,
               config['training_data']['input_transform']['batch_size'] // 2),
        axis=0,
    )
    diff = model.net.Sub(
        [left, right],
        ['diff'],
        broadcast=1,
        axis=0,
    )
    diff_abs = model.net.Abs(diff, 'diff_abs')
    diff_square = model.net.Mul(
        [diff_abs, diff_abs],
        ['diff_square'],
        broadcast=1,
        axis=0,
    )
    diff_square_sum = model.net.ReduceBackSum(
        [diff_square],
        ['diff_square_sum'],
    )
    diff_sqrt = model.net.Sqrt(
        [diff_square_sum],
        ['diff_sqrt'],
    )
    diff_norm2 = model.net.ReduceBackSum(
        [diff_sqrt],
        ['diff_norm2'],
    )
    BATCHSIZE = model.net.GivenTensorFill(
        [],
        ['BATCHSIZE'],
        values=np.asarray(config[
            'training_data']['input_transform']['batch_size']).astype(np.float32),
        shape=(),
    )
    pc_reg = model.net.Div(
        [diff_norm2, BATCHSIZE],
        ['pc_reg'],
        # broadcast=1,
        # axis=0,
    )

    # sum the softmax-loss with pc-reg
    pc_weight = model.net.ConstantFill(
        [],
        ['pc_weight'],
        value=config['solver']['pc_weight'],
    )
    weighted_pc_reg = model.net.Mul(
        [pc_reg, pc_weight],
        ['weighted_pc_reg'],
        broadcast=1,
        axis=0,
    )
    loss = model.net.Sum(
        [softmax_loss, weighted_pc_reg],
        ['loss'],
    )

    return loss


def add_optimizer(model, config):
    optimizer.add_weight_decay(model, config['solver']['weight_decay'])
    optimizer.build_multi_precision_sgd(
        model,
        base_learning_rate = config['solver']['base_learning_rate'],
        momentum = config['solver']['momentum'],
        nesterov = config['solver']['nesterov'],
        policy = config['solver']['lr_policy'],
        power = config['solver']['power'],
        max_iter = config['solver']['max_iter'],
        # gamma = config['solver']['gamma'],
        # stepsize = config['solver']['stepsize'],
    )


def add_training_operators(model, config, loss):
    """
    compute model loss and add backword propagation with optimization method
    """
    model.AddGradientOperators([loss])
    add_optimizer(model, config)


def add_accuracy(model):
    """ compute model classification accuracy """
    accuracy = brew.accuracy(
        model,
        ['softmax', 'label'],
        "accuracy"
    )
    accuracy_5 = model.net.Accuracy(
        ['softmax', 'label'],
        "accuracy_5",
        top_k=5,
    )
    return (accuracy, accuracy_5)



if __name__ == '__main__':
    epoch_results = [1,2,3,4,5,6]
    config = dict()
    config['root_dir'] = '/home/zhibin/wangxiao/workshop/fgvc-tasks/Bi-Attention/'
    dst_path = ''
    name = 'hola'
    postfix = 'test'
    color = 'r'
    shape = '.'

    plot_history(epoch_results, config, dst_path, name, postfix, color, shape)

