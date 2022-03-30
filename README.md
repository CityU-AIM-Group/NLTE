# Towards Robust Adaptive Object Detection under Noisy Annotations
Code for CVPR-22 paper "[Towards Robust Adaptive Object Detection under Noisy Annotations](https://github.com/CityU-AIM-Group/NLTE)"

## 1. Introduction
Existing methods assume that the source domain labels are completely clean, yet large-scale datasets often contain error-prone annotations due to instance ambiguity, which may lead to a biased source distribution and severely degrade the performance of the domain adaptive detector de facto.

## 2. Data Preparation

Prepare VOC data set following [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)

Prepare Clipart/Watercolor data set following [cross-domain-detection](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets)

Prepare Cityscapes/Foggy Cityscapes data set following [Everypixelmatters](https://github.com/chengchunhsu/EveryPixelMatters#dataset)

Change the directory in ./maskrcnn_benchmark/config/path_catalog.py correspondingly.

## 3. Installation (following [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark))

### Requirements:
- PyTorch 1.0 from a nightly release. Installation instructions can be found in https://pytorch.org/get-started/locally/
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo


### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name nlte
source activate nlte

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install pytorch-nightly -c pytorch

# install torchvision
cd ~/github
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install

# install pycocotools
cd ~/github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd ~/github
git clone https://github.com/CityU-AIM-Group/NLTE.git
cd NLTE
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```

## 4. Generate Symmetric Noise for Source domain (VOC)

Put make_sym_noise.py into VOCdevkit, the structure should be like:
```
VOCdevkit
└─ VOC2007
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
└─ VOC2012
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
└─ make_sym_noise.py
```
run 
```
python make_sym_noise.py
```

## 5. Train/test The Model

Train:
```
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/train_net.py --config-file "path/to/config/file.yaml" DATASETS.ANNOTATIONS Noisy_Annotations_per_$NOISYRATE
```
Test:
```
python /path_to_maskrcnn_benchmark/tools/train_net.py --config-file "path/to/config/file.yaml" MODEL.WEIGHT $YOURWEIGHT
```
The trained models are provided in [Google Drive](https://drive.google.com/drive/folders/1Yq4lPTH2cIs5156uLMItTBa8kA-YZ34I?usp=sharing)
                                      
## 6. Citation

If you found this work useful for your research, please cite our paper:
```
@inproceedings{liu2022towards,
  title={Towards Robust Adaptive Object Detection under Noisy Annotations},
  author={Liu, Xinyu and Li, Wuyang and Yang, Qiushi and Li, Baopu and Yuan, Yixuan},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## 7. Acknowledgement

This work is based on [Domain-Adaptive-Faster-RCNN](https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
