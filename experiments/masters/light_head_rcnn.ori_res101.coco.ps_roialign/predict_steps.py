# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IPython import embed
from config import cfg, config

import time

import uuid
import argparse
import dataset
import os.path as osp
import network_desp
import tensorflow as tf
import numpy as np
import cv2, os, sys, math, json, pickle
from PIL import Image

from tqdm import tqdm
from utils.py_faster_rcnn_utils.cython_nms import nms, nms_new
from utils.py_utils import misc

from multiprocessing import Queue, Process
from detection_opr.box_utils.box import DetBox
from detection_opr.utils.bbox_transform import clip_boxes, bbox_transform_inv
from functools import partial


def load_model(model_file, dev):
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    net = network_desp.Network()
    inputs = net.get_inputs()
    net.inference('PREDICT', inputs)
    test_collect_dict = net.get_test_collection()
    test_collect = [it for it in test_collect_dict.values()]
    saver = tf.train.Saver()

    saver.restore(sess, model_file)
    return partial(sess.run, test_collect), inputs


def inference(val_func, inputs, data_dict):
    image = data_dict
    ori_shape = image.shape

    if config.eval_resize == False:
        resized_img, scale = image, 1
    else:
        resized_img, scale = dataset.resize_img_by_short_and_max_size(
            image, config.eval_image_short_size, config.eval_image_max_size)
    height, width = resized_img.shape[0:2]

    resized_img = resized_img.astype(np.float32) - config.image_mean
    resized_img = np.ascontiguousarray(resized_img[:, :, [2, 1, 0]])

    im_info = np.array(
        [[height, width, scale, ori_shape[0], ori_shape[1], 0]],
        dtype=np.float32)

    feed_dict = {inputs[0]: resized_img[None, :, :, :], inputs[1]: im_info}

    _, scores, pred_boxes, rois = val_func(feed_dict=feed_dict)



    boxes = rois[:, 1:5] / scale

    if cfg.TEST.BBOX_REG:
        pred_boxes = bbox_transform_inv(boxes, pred_boxes)
        pred_boxes = clip_boxes(pred_boxes, ori_shape)

    pred_boxes = pred_boxes.reshape(-1, config.num_classes, 4)
    result_boxes = []

    for j in range(1, config.num_classes):
        inds = np.where(scores[:, j] > config.test_cls_threshold)[0]
        cls_scores = scores[inds, j]
        cls_bboxes = pred_boxes[inds, j, :]
        cls_dets = np.hstack((cls_bboxes, cls_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        keep = nms(cls_dets, config.test_nms)
        cls_dets = np.array(cls_dets[keep, :], dtype=np.float, copy=False)
        for i in range(cls_dets.shape[0]):
            db = cls_dets[i, :]
            dbox = DetBox(
                db[0], db[1], db[2] - db[0], db[3] - db[1],
                tag=config.class_names[j], score=db[-1])
            result_boxes.append(dbox)

    if len(result_boxes) > config.test_max_boxes_per_image:
        result_boxes = sorted(
            result_boxes, reverse=True, key=lambda t_res: t_res.score) \
            [:config.test_max_boxes_per_image]
    result_dict = dict()
    result_dict['result_boxes'] = result_boxes
    return result_dict


def worker(model_file, dev, records, read_func, result_queue):
    func, inputs = load_model(model_file, dev)
    for record in records:
        data_dict = read_func(record)
        result_dict = inference(func, inputs, data_dict)
        result_queue.put_nowait(result_dict)


def predict_video(args):
    devs = args.devices.split(',')
    misc.ensure_dir(config.eval_dir)
    #vidpath = 'http://192.168.1.137:8080/video' #Home IP Camera
    vidpath = 'http://10.42.0.37:8080/video' #Lab IP Camera
    f = open("step-by-step_guide.txt", "r")
    steps = [x.split('\n') for x in f.read().split("#") if x != '']
    for i in range(len(steps)):
        steps[i] = [x for x in steps[i] if x != '']
    m = open("motherboard_database.txt", "r")
    mb_info = [x.split('#') for x in m.read().split("\n") if x != '']
    current_mb = []
    cap = cv2.VideoCapture(vidpath)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    font = cv2.FONT_ITALIC
    step = 0
    epoch_num = args.predict_epoch
    model_file = osp.join( config.output_dir, 'model_dump','epoch_{:d}'.format(epoch_num) + '.ckpt')
    func, inputs = load_model(model_file, devs[0])
    avg_fps = []
    thresh = 0.4
    thresh_list = {2:0.6,4:0.4,5:0.4,8:0.8,9:0.4,11:0.6,12:0.4,14:0.4,15:0.4,17:0.5,18:0.4,19:0.4,20:0.7,22:0.2,24:0.4,25:0.4,27:0.4,28:0.4,30:0.8,31:0.95,33:0.7,34:0.4,36:0.7,37:0.4}
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #out = cv2.VideoWriter('/home/erik/Documents/light_head_rcnn-master/data/motherboard/test/test_to_result/output_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))
    while(True):
        #fps_start = time.time()
        ret, frame = cap.read()
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        if(ret == True and step < len(steps)):
            draw_steps(steps, step, frame, w, h, font,current_mb)
            specify_classes = get_specific_classes(steps,step)
            if specify_classes != 0:
                result_dict = inference(func, inputs, frame)
                if type(specify_classes) == str:
                    specify_classes = [specify_classes]
                if step in thresh_list.keys():
                    thresh = thresh_list[step]
                else:
                    thresh = 0.4
                for db in result_dict['result_boxes']:
                    if db.score > thresh and db.tag in specify_classes: #preferred score of box to apply to picture
                        #print(db)
                        db.draw(frame)
                        if db.tag in config.class_names[1:3+1]:
                            for i in mb_info:
                                if db.tag == i[0]:
                                    current_mb = i
                            
                cv2.imshow('image', frame) #display results'
                #out.write(frame)
            else:
                cv2.imshow('image', frame)  # display results'
                #out.write(frame)
            #fps_end = time.time()
            #frame_rate = 1 / (fps_end - fps_start)
            #print("FPS: ",frame_rate)
            #print(thresh)
            #avg_fps.append(frame_rate)
            k = cv2.waitKey(1)
            if k:
                if k == ord(' '): #Click 'space' to go to next step
                    step += 1
                elif k == ord('i'): #increase threshold
                    if thresh < 1.0:
                        thresh += 0.05
                elif k == ord('d'):
                    if thresh > 0.1: #decrease threshold
                        thresh -= 0.05
                elif k == ord('p') and step > 0: #Click 'p' to go to previous step if step > 0
                    step -= 1
                elif k == ord('q'): #Click 'q' to stop the guide
                    print("The step-by-step guide interrupted...")
                    break

        else:
            break
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    #total_fps = sum(avg_fps)/len(avg_fps)
    #print("AVERAGE FPS: ",total_fps)
    print("\n")

def draw_steps(steps, step, frame, w, h, font,current_mb):
    increment = 15
    
    non_black_screen = [2,4,5,8,9,11,12,14,15,17,18,19,20,22,24,25,27,28,30,31,33,34,36,37]
    if step in non_black_screen:
        black_background = len(steps[step])*10+10
        cv2.rectangle(frame, (0, black_background), (int(w), 0), (0, 0, 0), -1)
        if step == 4 and len(current_mb) > 0:
            for i in steps[step]:
                if i == "Supported RAM for this motherboard:":
                    i = i+" DDR"+current_mb[-1]
                    cv2.putText(frame, i, (5, increment), font, 1 / 3, (255, 255, 255), 1, cv2.LINE_AA)
                    increment += 10
                else:
                    cv2.putText(frame, i, (5, increment), font, 1 / 3, (255, 255, 255), 1, cv2.LINE_AA)
                    increment += 10
        else:
            for i in steps[step]:
                cv2.putText(frame, i, (5, increment), font, 1 / 3, (255, 255, 255), 1, cv2.LINE_AA)
                increment += 10
    elif step == 7 and len(current_mb) > 0:
        cv2.rectangle(frame, (0, int(h)), (int(w), 0), (0, 0, 0), -1)
        for i in steps[step]:
            if i == "Generation/Support:":
                i = i+" "+current_mb[3] #Generation for the CPU based on the motherboard
                cv2.putText(frame, i, (5, increment), font, 1 / 3, (255, 255, 255), 1, cv2.LINE_AA)
                increment += 10
            elif i == "Socket type:":
                i = i+" "+current_mb[1] #Socket type for the CPU based on the motherboard
                cv2.putText(frame, i, (5, increment), font, 1 / 3, (255, 255, 255), 1, cv2.LINE_AA)
                increment += 10
            else:
                cv2.putText(frame, i, (5, increment), font, 1 / 3, (255, 255, 255), 1, cv2.LINE_AA)
                increment += 10
    else:
        cv2.rectangle(frame, (0, int(h)), (int(w), 0), (0, 0, 0), -1)
        for i in steps[step]:
            cv2.putText(frame, i, (5, increment), font, 1 / 3, (255, 255, 255), 1, cv2.LINE_AA)
            increment += 10

def get_specific_classes(steps, step ):
    '''if step == 0 or step == 4 or step == 5 or step == (len(steps)-1):
        return 0'''
    if step == 2:
        return config.class_names[1:3+1]
    elif step == 4:
        return config.class_names[17:18+1]
    elif step == 5:
        return config.class_names[14]
    elif step == 8:
        return config.class_names[21]
    elif step == 9:
        return config.class_names[5]
    elif step == 11:
        return config.class_names[22]
    elif step == 12:
        return config.class_names[11]
    elif step == 14:
        return config.class_names[12]
    elif step == 15:
        return config.class_names[23]
    elif step == 17:
        return config.class_names[24]
    elif step == 18:
        return config.class_names[25:26+1]
    elif step == 19:
        return config.class_names[20]
    elif step == 20:
        return config.class_names[4]
    elif step == 22:
        return config.class_names[6:7+1]
    elif step == 24:
        return config.class_names[28]
    elif step == 25:
        return config.class_names[9]
    elif step == 27:
        return config.class_names[27]
    elif step == 28:
        return config.class_names[8]
    elif step == 30:
        return config.class_names[29]
    elif step == 31 or step == 34:
        return config.class_names[10]
    elif step == 33:
        return config.class_names[15:16+1]
    elif step == 36:
        return config.class_names[19]
    elif step == 37:
        return config.class_names[13]
    else:
        return 0


def make_parser():
    parser = argparse.ArgumentParser('test network')
    parser.add_argument('-d', '--devices', default='0', type=str, help='device for testing')
    parser.add_argument('--predict_epoch', '-pe', default=1, type=int, help='epoch to use for prediction')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    args.devices = misc.parse_devices(args.devices)
    predict_video(args)
