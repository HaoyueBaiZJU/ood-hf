# Out-of-Distribution Learning with Human Feedback


### Abstract

Out-of-distribution (OOD) learning often relies on strong statistical assumptions or predefined OOD data distributions, limiting its effectiveness in real-world deployment for both OOD generalization and detection, especially when human inspection is minimal. This paper introduces a novel framework for OOD learning that integrates human feedback to enhance model adaptation and reliability. Our approach leverages freely available unlabeled data in the wild, which naturally captures environmental test-time OOD distributions under both covariate and semantic shifts. To effectively utilize such data, we propose selectively acquiring human feedback to label a small subset of informative samples. These labeled samples are then used to train both a multi-class classifier and an OOD detector. By incorporating human feedback, our method significantly improves model robustness and precision in handling OOD scenarios. We provide theoretical insights by establishing generalization error bounds for our algorithm. Extensive experiments demonstrate that our approach outperforms state-of-the-art methods by a significant margin.

## Quick Start

### Data Preparation
In this work, we evaluate the OOD generalization and detection performance over a range of environmental discrepancies such as domains, image corruptions, and perturbations. 

Download the data in the folder

```
./datasets
```



#### CIFAR-10 & CIFAR-10-C

- Create a folder named `cifar-10/` and a folder `cifar-10-c/` under `$datasets`.
- Download the dataset from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) and extract the training and validation sets to `$DATA/cifar-10/`.
- Refer the dataset from the [CIFAR-10-C](https://arxiv.org/abs/1903.12261) and extract the training and test sets to `$DATA/cifar-10-c/`. The directory structure should look like


The corrupted CIFAR-10 dataset can be downloaded via the link:
```
wget https://drive.google.com/drive/u/0/folders/1JcI8UMBpdMffzCe-dqrzXA9bSaEGItzo
```


```
cifar-10/
|–– train/ 
|–– val/
cifar-10-c/
|–– CorCIFAR10_train/ 
|–– CorCIFAR10_test/
```


Here are links for the less common semantic OOD datasets regarding CIFAR benchmark used in the paper: 
[Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/),
[Places365](http://places2.csail.mit.edu/download.html), 
[LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz),
[LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz),
[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz).

For example, run the following commands in the **root** directory to download **LSUN-C**:
```
cd data/LSUN
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```



## Training and Evaluation 

**Pretrained models**

You can find the pretrained models in 

```
./checkpoints/Resnet34_vanilla.pt
./checkpoints/cifar10_wrn_pretrained.pt
./checkpoints/cifar100_wrn_pretrained.pt
```

**The setected data for ImageNet-100 benchmark**

* ```selected_category_imagenet.pt```
* ```selected_data_imagenet.pt```
* ```selected_label_imagenet.pt```

**Demo** 

We provide sample scripts to run the code. Feel free to modify the hyperparameters and training configurations.

```
bash run.sh
```

### Citation

If you find our work useful, please consider citing our paper:

```
@article{bai2024out,
  title={Out-of-Distribution Learning with Human Feedback},
  author={Bai, Haoyue and Du, Xuefeng and Rainey, Katie and Parameswaran, Shibin and Li, Yixuan},
  journal={arXiv preprint arXiv:2408.07772},
  year={2024}
}
```

s
