# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IPython import embed
from config import cfg, config
from flask import Flask, render_template, Response,request

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


class predict_video(object):
    def __init__(self):
        #self.vidpath = 'http://192.168.1.137:8080/video' #Home IP Camera
        self.vidpath = 'http://10.42.0.37:8080/video' #Lab IP Camera
        #vidpath = '/home/erik/Downloads/20190310_085513.mp4'
        #vidpath = '/home/jovyan/test_video.mp4'
        self.f = open("step-by-step_guide.txt", "r")
        self.steps = [x.split('\n') for x in self.f.read().split("#") if x != '']
        for i in range(len(self.steps)):
            self.steps[i] = [x for x in self.steps[i] if x != '']
        self.cap = cv2.VideoCapture(self.vidpath)
        self.w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.font = cv2.FONT_ITALIC
        self.step = 0
        self.epoch_num = 29
        self.model_file = osp.join(config.output_dir, 'model_dump','epoch_{:d}'.format(self.epoch_num) + '.ckpt')
        self.func, self.inputs = load_model(self.model_file, '0')
    def get_frame(self, key = 'x'):
        if key == 'n' and self.step < 38:
            self.step += 1
        elif key == 'p' and self.step > 0:
            self.step -= 1
        elif key == 'q':
            self.step = 38
        thresh = 0.4
        ret, frame = self.cap.read()
        if(ret == True):
            draw_steps(self.steps, self.step, frame, self.w, self.h, self.font)
            specify_classes = get_specific_classes(self.steps,self.step)
            if specify_classes != 0:
                result_dict = inference(self.func, self.inputs, frame)
                if type(specify_classes) == str:
                    specify_classes = [specify_classes]
                for db in result_dict['result_boxes']:
                   if db.score > thresh and db.tag in specify_classes:
                        db.draw(frame)
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
            else:
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()

def draw_steps(steps, step, frame, w, h, font):
    increment = 15
    
    non_black_screen = [2,4,5,8,9,11,12,14,15,17,18,19,20,22,24,25,27,28,30,31,33,34,36,37]
    if step in non_black_screen:
        black_background = len(steps[step])*10+10
        cv2.rectangle(frame, (0, black_background), (int(w), 0), (0, 0, 0), -1)
        for i in steps[step]:
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


'''def make_parser():
    parser = argparse.ArgumentParser('test network')
    parser.add_argument('-d', '--devices', default='0', type=str, help='device for testing')
    parser.add_argument('--predict_epoch', '-pe', default=29, type=int, help='epoch to use for prediction')
    return parser'''

app = Flask(__name__)
preview = predict_video()
data = ""
@app.route('/',methods=['GET','POST'])
def index():
    global data
    data = request.args.get("key")
    return render_template('index.html')

def gen(camera):
    while True:
        global data
        if data != 'x':
            frame = camera.get_frame(data)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            data = 'x'
        else:
            frame = camera.get_frame()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            

@app.route('/video_feed')
def video_feed():
    return Response(gen(preview), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    #parser = make_parser()
    #args = parser.parse_args()
    #args.devices = misc.parse_devices(args.devices)

    app.run(host='0.0.0.0')
    #predict_video(args)

