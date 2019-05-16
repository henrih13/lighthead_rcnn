# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

from IPython import embed
from config import cfg
import tensorflow as tf
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim.python.slim.nets import resnet_utils, resnet_v1
import numpy as np

from utils.tf_utils.basemodel import resnet_utils, resnet_v1 ##implement Xception instead
from utils.tf_utils.basemodel import xception
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import regularizers, \
    initializers, layers

# todo : delete snippets
from detection_opr.rpn.snippets import generate_anchors_opr
from detection_opr.rpn_batched.proposal_target_layer import proposal_target_layer
from detection_opr.rpn_batched.anchor_target_layer_without_boxweight import \
    anchor_target_layer
from detection_opr.rpn_batched.proposal_opr import proposal_opr

from detection_opr.utils import loss_opr
from lib_kernel.lib_psroi_pooling import psroi_pooling_op, psroi_pooling_op_grad
from lib_kernel.lib_psalign_pooling import psalign_pooling_op, psalign_pooling_op_grad
from detection_opr.rfcn_plus_plus import rfcn_plus_plus_opr
from collections import OrderedDict as dict


''''def resnet_arg_scope(
        is_training=True, weight_decay=cfg.weight_decay, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': cfg.bn_training,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=initializers.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc'''


def xception_arg_scope(weight_decay=0.00001,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001):
  '''
  The arg scope for xception model. The weight decay is 1e-5 as seen in the paper.

  INPUTS:
  - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
  - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
  - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.

  OUTPUTS:
  - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
  '''
  # Set weight_decay for weights in conv2d and separable_conv2d layers.
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=None,
                      activation_fn=None):
            
    # Set parameters for batch_norm. Note: Do not set activation function as it's preset to None already.
    with slim.arg_scope([slim.batch_norm],
                        decay=batch_norm_decay,
                        epsilon=batch_norm_epsilon) as scope:
      return scope



class Network(object):
    def __init__(self):
        pass

    def get_inputs(self, mode=0):
        """gets network inputs
        Returns:
             mode:
                0: return inputs tensor_holder
                1: return inputs name
        """
        if mode == 0:
            inputs = []
            inputs.append(tf.placeholder(tf.float32, shape=[None, None, None, 3]))
            inputs.append(tf.placeholder(tf.float32, shape=[None, 6]))
            inputs.append(tf.placeholder(tf.float32, shape=[None, None, 5]))
            return inputs
        elif mode == 1:
            inputs_names = ['data', 'im_info', 'boxes']
            return inputs_names

    def inference(self, mode, inputs):
        is_training = mode == 'TRAIN'

        ###decode your inputs
        [image, im_info, gt_boxes] = inputs

        image.set_shape([None, None, None, 3])
        im_info.set_shape([None, cfg.nr_info_dim])
        if mode == 'TRAIN':
            gt_boxes.set_shape([None, None, 5])
        ##end of decode

        num_anchors = len(cfg.anchor_scales) * len(cfg.anchor_ratios)
        #bottleneck = resnet_v1.bottleneck
        #xception_module = xception.xception_module

        '''blocks = [
            resnet_utils.Block('block1', bottleneck,
                               [(256, 64, 1, 1)] * 2 + [(256, 64, 1, 1)]),
            resnet_utils.Block('block2', bottleneck,
                               [(512, 128, 2, 1)] + [(512, 128, 1, 1)] * 3),
            resnet_utils.Block('block3', bottleneck,
                               [(1024, 256, 2, 1)] + [(1024, 256, 1, 1)] * 22),
            resnet_utils.Block('block4', bottleneck,
                               [(2048, 512, 1, 2)] + [(2048, 512, 1, 2)] * 2)
        ]'''






        '''with slim.arg_scope(resnet_arg_scope(is_training=False)):
            with tf.variable_scope('resnet_v1_101', 'resnet_v1_101'):
                net = resnet_utils.conv2d_same(
                    image, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(
                    net, [3, 3], stride=2, padding='SAME', scope='pool1')
            net, _ = resnet_v1.resnet_v1(
                net, blocks[0:1], global_pool=False, include_root_block=False,
                scope='resnet_v1_101')
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net_conv3, _ = resnet_v1.resnet_v1(
                net, blocks[1:2], global_pool=False, include_root_block=False,
                scope='resnet_v1_101')
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net_conv4, _ = resnet_v1.resnet_v1(
                net_conv3, blocks[2:3], global_pool=False,
                include_root_block=False, scope='resnet_v1_101')
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net_conv5, _ = resnet_v1.resnet_v1(
                net_conv4, blocks[-1:], global_pool=False,
                include_root_block=False, scope='resnet_v1_101')'''


        with tf.variable_scope('Xception') as sc:
            end_points_collection = sc.name + '_end_points'
        
            with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1),\
              slim.arg_scope([slim.separable_conv2d, slim.conv2d, slim.avg_pool2d], outputs_collections=[end_points_collection]),\
              slim.arg_scope([slim.batch_norm], is_training=is_training):

                #===========ENTRY FLOW==============
                #Block 1
                net = slim.conv2d(image, 32, [3,3], stride=2, padding='valid', scope='block1_conv1')
                net = slim.batch_norm(net, scope='block1_bn1')
                net = tf.nn.relu(net, name='block1_relu1')
                net = slim.conv2d(net, 64, [3,3], padding='valid', scope='block1_conv2')
                net = slim.batch_norm(net, scope='block1_bn2')
                net = tf.nn.relu(net, name='block1_relu2')
                residual = slim.conv2d(net, 128, [1,1], stride=2, scope='block1_res_conv')
                residual = slim.batch_norm(residual, scope='block1_res_bn')
    
                #Block 2
                net = slim.separable_conv2d(net, 128, [3,3], scope='block2_dws_conv1')
                net = slim.batch_norm(net, scope='block2_bn1')
                net = tf.nn.relu(net, name='block2_relu1')
                net = slim.separable_conv2d(net, 128, [3,3], scope='block2_dws_conv2')
                net = slim.batch_norm(net, scope='block2_bn2')
                net = slim.max_pool2d(net, [3,3], stride=2, padding='same', scope='block2_max_pool')
                net = tf.add(net, residual, name='block2_add')
                residual = slim.conv2d(net, 256, [1,1], stride=2, scope='block2_res_conv')
                residual = slim.batch_norm(residual, scope='block2_res_bn')
    
                #Block 3
                net = tf.nn.relu(net, name='block3_relu1')
                net = slim.separable_conv2d(net, 256, [3,3], scope='block3_dws_conv1')
                net = slim.batch_norm(net, scope='block3_bn1')
                net = tf.nn.relu(net, name='block3_relu2')
                net = slim.separable_conv2d(net, 256, [3,3], scope='block3_dws_conv2')
                net = slim.batch_norm(net, scope='block3_bn2')
                net = slim.max_pool2d(net, [3,3], stride=2, padding='same', scope='block3_max_pool')
                net = tf.add(net, residual, name='block3_add')
                residual = slim.conv2d(net, 728, [1,1], stride=2, scope='block3_res_conv')
                residual = slim.batch_norm(residual, scope='block3_res_bn')
    
                #Block 4
                net = tf.nn.relu(net, name='block4_relu1')
                net = slim.separable_conv2d(net, 728, [3,3], scope='block4_dws_conv1')
                net = slim.batch_norm(net, scope='block4_bn1')
                net = tf.nn.relu(net, name='block4_relu2')
                net = slim.separable_conv2d(net, 728, [3,3], scope='block4_dws_conv2')
                net = slim.batch_norm(net, scope='block4_bn2')
                net = slim.max_pool2d(net, [3,3], stride=2, padding='same', scope='block4_max_pool')
                net = tf.add(net, residual, name='block4_add')
    
                
    
    
    
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        

        '''with tf.variable_scope(
                'resnet_v1_101', 'resnet_v1_101',
                regularizer=tf.contrib.layers.l2_regularizer(
                    cfg.weight_decay)):
            # rpn
            rpn = slim.conv2d(
                net_conv4, 512, [3, 3], trainable=is_training,  ##change to 256 for Xception
                weights_initializer=initializer, activation_fn=nn_ops.relu,
                scope="rpn_conv/3x3")
            rpn_cls_score = slim.conv2d(
                rpn, num_anchors * 2, [1, 1], trainable=is_training,
                weights_initializer=initializer, padding='VALID',
                activation_fn=None, scope='rpn_cls_score')
            rpn_bbox_pred = slim.conv2d(
                rpn, num_anchors * 4, [1, 1], trainable=is_training,
                weights_initializer=initializer, padding='VALID',
                activation_fn=None, scope='rpn_bbox_pred')'''

        with tf.variable_scope(
                'Xception',
                regularizer=tf.contrib.layers.l2_regularizer(
                    cfg.weight_decay)):
            # rpn
            rpn = slim.conv2d(
                net, 256, [3, 3], trainable=is_training,  ##change to 256 for Xception
                weights_initializer=initializer, activation_fn=nn_ops.relu,
                scope="rpn_conv/3x3")
            rpn_cls_score = slim.conv2d(
                rpn, num_anchors * 2, [1, 1], trainable=is_training,
                weights_initializer=initializer, padding='VALID',
                activation_fn=None, scope='rpn_cls_score')
            rpn_bbox_pred = slim.conv2d(
                rpn, num_anchors * 4, [1, 1], trainable=is_training,
                weights_initializer=initializer, padding='VALID',
                activation_fn=None, scope='rpn_bbox_pred')

            # generate anchor
            height = tf.cast(tf.shape(rpn)[1], tf.float32)
            width = tf.cast(tf.shape(rpn)[2], tf.float32)
            anchors = generate_anchors_opr(
                height, width, cfg.stride[0], cfg.anchor_scales,
                cfg.anchor_ratios)
            # change it so that the score has 2 as its channel size
            rpn_cls_prob = tf.reshape(rpn_cls_score, [-1, 2])
            rpn_cls_prob = tf.nn.softmax(rpn_cls_prob, name='rpn_cls_prob')
            rpn_cls_prob = tf.reshape(rpn_cls_prob, tf.shape(rpn_cls_score))

            rois, roi_scores = proposal_opr(
                rpn_cls_prob, rpn_bbox_pred, im_info, mode, cfg.stride,
                anchors, num_anchors, is_tfchannel=True, is_tfnms=True)

            if is_training:
                with tf.variable_scope('anchor') as scope:
                    rpn_labels, rpn_bbox_targets = \
                        tf.py_func(
                            anchor_target_layer,
                            [gt_boxes, im_info, cfg.stride, anchors,
                             num_anchors],
                            [tf.float32, tf.float32])
                    rpn_labels = tf.to_int32(rpn_labels, name="to_int32")

                with tf.control_dependencies([rpn_labels]):
                    with tf.variable_scope('rpn_rois') as scope:
                        rois, labels, bbox_targets = \
                            tf.py_func(
                                proposal_target_layer,
                                [rois, gt_boxes, im_info],
                                [tf.float32, tf.float32, tf.float32])
                        labels = tf.to_int32(labels, name="to_int32")


        '''with tf.variable_scope(
                'resnet_v1_101', 'resnet_v1_101',
                regularizer=tf.contrib.layers.l2_regularizer(
                    cfg.weight_decay)):'''

            
        with tf.variable_scope(
                'Xception',
                regularizer=tf.contrib.layers.l2_regularizer(
                    cfg.weight_decay)):
            ps_chl = 7 * 7 * 10
            ps_fm = rfcn_plus_plus_opr.global_context_module(
                net, prefix='conv_new_1',
                ks=15, chl_mid=256, chl_out=ps_chl)
            ps_fm = nn_ops.relu(ps_fm)

            [psroipooled_rois, _, _] =  psalign_pooling_op.psalign_pool(
                ps_fm, rois, group_size=7,
                sample_height=2, sample_width=2, spatial_scale=1.0/16.0)

            #[psroipooled_rois, _] = psroi_pooling_op.psroi_pool(
            #    ps_fm, rois, group_size=7, spatial_scale=1.0 / 16.0)
            psroipooled_rois = slim.flatten(psroipooled_rois)
            ps_fc_1 = slim.fully_connected(
                psroipooled_rois, 2048, weights_initializer=initializer,
                activation_fn=nn_ops.relu, trainable=is_training, scope='ps_fc_1')
            cls_score = slim.fully_connected(
                ps_fc_1, cfg.num_classes, weights_initializer=initializer,
                activation_fn=None, trainable=is_training, scope='cls_fc')
            bbox_pred = slim.fully_connected(
                ps_fc_1, 4 * cfg.num_classes, weights_initializer=initializer_bbox,
                activation_fn=None, trainable=is_training, scope='bbox_fc')

            cls_prob = loss_opr.softmax_layer(cls_score, "cls_prob")

            #conv_new_1 = slim.conv2d(
            #    net_conv5, 1024, [1, 1], trainable=is_training,
            #    weights_initializer=initializer, activation_fn=nn_ops.relu,
            #    scope="conv_new_1")
            #rfcn_cls = slim.conv2d(
            #    conv_new_1, 7 * 7 * cfg.num_classes, [1, 1],
            #    trainable=is_training, weights_initializer=initializer,
            #    activation_fn=None, scope="rfcn_cls")
            #rfcn_bbox = slim.conv2d(
            #    conv_new_1, 7 * 7 * 4, [1, 1], trainable=is_training,
            #    weights_initializer=initializer,
            #    activation_fn=None, scope="rfcn_bbox")

            #[psroipooled_cls_rois, _] = psroi_pooling_op.psroi_pool(
            #    rfcn_cls, rois, group_size=7, spatial_scale=1.0 / 16.0)
            #[psroipooled_loc_rois, _] = psroi_pooling_op.psroi_pool(
            #    rfcn_bbox, rois, group_size=7, spatial_scale=1.0 / 16.0)

            #cls_score = tf.reduce_mean(psroipooled_cls_rois, axis=[1, 2])
            #bbox_pred = tf.reduce_mean(psroipooled_loc_rois, axis=[1, 2])
            #cls_prob = loss_opr.softmax_layer(cls_score, "cls_prob")
            # cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
            #bbox_pred = tf.tile(bbox_pred, [1, cfg.num_classes])

        if not is_training:
            stds = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (cfg.num_classes))
            means = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (cfg.num_classes))
            bbox_pred *= stds
            bbox_pred += means

            ##############add prediction#####################
            tf.add_to_collection("rpn_cls_score", rpn_cls_score)
            tf.add_to_collection("rpn_cls_prob", rpn_cls_prob)
            tf.add_to_collection("rpn_bbox_pred", rpn_bbox_pred)
            tf.add_to_collection("cls_score", cls_score)
            tf.add_to_collection("cls_prob", cls_prob)
            tf.add_to_collection("bbox_pred", bbox_pred)
            tf.add_to_collection("rois", rois)

        else:
            #--------------------  rpn loss ---------------------------------#
            from detection_opr.utils import loss_opr_without_box_weight
            rpn_loss_box = loss_opr_without_box_weight.smooth_l1_loss_rpn(
                tf.reshape(rpn_bbox_pred, [-1, 4]),
                tf.reshape(rpn_bbox_targets, [-1, 4]),
                tf.reshape(rpn_labels, [-1]), sigma=cfg.simga_rpn)

            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
            rpn_label = tf.reshape(rpn_labels, [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(
                tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=rpn_cls_score, labels=rpn_label))

            #-------------------- rcnn loss  --------------------------------#
            label = tf.reshape(labels, [-1])
            cross_entropy, loss_box = loss_opr_without_box_weight.sum_ohem_loss(
                tf.reshape(cls_score, [-1, cfg.num_classes]), label,
                bbox_pred, bbox_targets, cfg.TRAIN.nr_ohem_sampling,
                cfg.num_classes)
            loss_box *= 2

            #--------------------add to colloection ------------------------#
            tf.add_to_collection('loss_cls', cross_entropy)
            tf.add_to_collection('loss_box', loss_box)
            tf.add_to_collection('rpn_loss_cls', rpn_cross_entropy)
            tf.add_to_collection('rpn_loss_box', rpn_loss_box)
            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            tf.add_to_collection('losses', loss)
            return loss

    def get_train_collection(self):
        ret = dict()
        ret['rpn_loss_cls'] = tf.add_n(tf.get_collection('rpn_loss_cls'))
        ret['rpn_loss_box'] = tf.add_n(tf.get_collection('rpn_loss_box'))
        ret['loss_cls'] = tf.add_n(tf.get_collection('loss_cls'))
        ret['loss_box'] = tf.add_n(tf.get_collection('loss_box'))
        ret['tot_losses'] = tf.add_n(tf.get_collection('losses'))
        return ret

    def get_test_collection(self):
        ret = dict()
        ret['cls_score'] = tf.get_collection('cls_score')[0]
        ret['cls_prob'] = tf.get_collection('cls_prob')[0]
        ret['bbox_pred'] = tf.get_collection('bbox_pred')[0]
        ret['rois'] = tf.get_collection('rois')[0]
        return ret
