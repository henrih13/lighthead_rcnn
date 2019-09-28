# Computer Vision based Auxiliary System for Computer Assembly: System Design and Implementation

## Requirements

Ubuntu 18.04<br />
Tensorflow 1.5.0<br />
Cuda 9.0<br />
cuDNN 7.0.5<br />

## Importing Backbone and Weights
1. Download and place [res101.ckpt](https://drive.google.com/open?id=1ISGXDyg5JUUX8NrekDTyRjuwH0E9qiy2) in 'data/imagenet_weights/'

2. Download and place both [epoch_29.ckpt.data-00000-of-00001](https://drive.google.com/open?id=1412Hyee1nGCQHvxXrHOed3haUTS7XhXv) and [epoch_29.ckpt.index](https://drive.google.com/open?id=1uDKZ9xqBlIihjLbZHNi92UEmkxyiqWM3) in 'output/erik/light_head_rcnn.ori_res101.coco.ps_roialign/model_dump/'  

## Citing Light-Head R-CNN

This implementation is based on the original Light-Head R-CNN paper.

## Preview

Youtube link:
[![Computer Vision based Auxiliary System for Computer Assembly: System Design and Implementation]](https://www.youtube.com/watch?v=B1MpH66SGIM&t=41s&fbclid=IwAR0fcFn4zsqhSjGuOPEsxx1clddyYGce29BB07qYoiWKOrDSKzMTJCTAfi0)

```
@article{li2017light,
  title={Light-Head R-CNN: In Defense of Two-Stage Object Detector},
  author={Li, Zeming and Peng, Chao and Yu, Gang and Zhang, Xiangyu and Deng, Yangdong and Sun, Jian},
  journal={arXiv preprint arXiv:1711.07264},
  year={2017}
}
```
