# Virtual Gram staining of label-free bacteria using CGANs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14232360.svg)](https://doi.org/10.5281/zenodo.14232360)

This repository contains the test code and trained checkpoints of [Virtual Gram staining of label-free bacteria using darkfield microscopy and deep learning](https://arxiv.org/abs/2407.12337).

A Conditional GAN was trained to perform virtual Gram staining of label-free bacteria.


<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="text-align: center;">
    <strong>Input images</strong><br>
    <img src="exp_1/test_images/2_22_inp_df_0min1plus1_2.jpg" width="200"/>
    <img src="exp_1/test_images/5_38_inp_df_0min1plus1_2.jpg" width="200"/>
    <!-- Add more input images as needed -->
  </div>
  <div style="text-align: center;">
    <strong>Output images</strong><br>
    <img src="exp_1/test_images/2_22_out.jpg" width="200"/>
    <img src="exp_1/test_images/5_38_out.jpg" width="200"/>

  </div>
  <div style="text-align: center;">
    <strong>Target images</strong><br>
    <img src="exp_1/test_images/2_22_tar.jpg" width="200"/>
    <img src="exp_1/test_images/5_38_tar.jpg" width="200"/>

  </div>
</div>


## How to start
* Download test data and checkpoints from this [link](https://drive.google.com/drive/folders/1f9eNcxyflmZJ7G47pdd6KyEzRdBxuTiU?usp=drive_link) or this [Zenodo link](https://zenodo.org/records/14232054)
* Move test_data and ckpts folders into the repository folder.
* Run test_npy.py file in your CPU or GPU.

## Requirements
* Tensorflow 2.10
* Python 3.9.17


## Support
* For any questions, please email cagatayisil@ucla.edu

