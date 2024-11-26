# Virtual Gram staining of bacteria using GANs
 Test code to run the trained virtual Gram staining model


This repository contains the test code and trained checkpoints of [Virtual Gram staining of label-free bacteria using darkfield microscopy and deep learning](https://arxiv.org/abs/2407.12337).

A Conditional GAN was trained to perform virtual Gram staining of label-free bacteria.

**Input images**
![img](exp_1/test_images/2_22_inp_df_0min1plus1_2.jpg)
**Output images**
![img](exp_1/test_images/2_22_out.jpg)
**Target images**
![img](exp_1/test_images/2_22_tar.jpg)

## How to start
* Download test data and checkpoints from this [link](https://drive.google.com/drive/folders/1f9eNcxyflmZJ7G47pdd6KyEzRdBxuTiU?usp=drive_link).
* Run test_npy.py file in your CPU or GPU

## Requirements
* Tensorflow 2.10
* Python 3.9.17