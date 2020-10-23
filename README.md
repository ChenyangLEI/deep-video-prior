# deep-video-prior (DVP)
Code for NeurIPS 2020 paper: Blind Video Temporal Consistency via Deep Video Prior



# CVPR2020-Polarized-Reflection-Removal-with-Perfect-Alignment
Code for CVPR2020 paper "Polarized Reflection Removal with Perfect Alignment in the Wild"

[paper](https://arxiv.org/abs/2010.11838)
| [project website](https://chenyanglei.github.io/DVP/index.html)


## Dependencey

### Environment
This code is based on tensorflow. It has been tested on Ubuntu 18.04 LTS.

Anaconda is recommended: [Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-18-04)
| [Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)

After installing Anaconda, you can setup the environment simply by

```
conda env create -f environment.yml
```

### Download checkpoint and VGG model

Download the VGG model from the [Google drive](https://drive.google.com/file/d/1Hm_VS5Acxge3t0TUhKFZa8c0QFw2TTr_/view?usp=sharing), put them in the correct path:

./VGG_Model/imagenet-vgg-verydeep-19.mat


## Inference

### Demo 
```
bash tesh.sh
```
The results are placed in ./result

### Use your own data 
For the video with unimodal inconsistency:

```
python main_IRT.py --max_epoch 25 --input PATH_TO_YOUR_INPUT_FOLDER --processed PATH_TO_YOUR_PROCESSED_FOLDER --model NAME_OF_YOUR_MODEL --with_IRT 0 --IRT_initialization 0 --output ./result/OWN_DATA
```

For the video with multimodal inconsistency:

```
python main_IRT.py --max_epoch 25 --input PATH_TO_YOUR_INPUT_FOLDER --processed PATH_TO_YOUR_PROCESSED_FOLDER --model NAME_OF_YOUR_MODEL --with_IRT 1 --IRT_initialization 1 --output ./result/OWN_DATA
```


## Citation
If you find this work useful for your research, please cite:
```
@inproceedings{lei2020dvp,
  title={Blind Video Temporal Consistency via Deep Video Prior},
  author={Lei, Chenyang and Xing, Yazhou and Chen, Qifeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}                
```

or 

```
@misc{lei2020blind,
      title={Blind Video Temporal Consistency via Deep Video Prior}, 
      author={Chenyang Lei and Yazhou Xing and Qifeng Chen},
      year={2020},
      eprint={2010.11838},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact
Please contact me if there is any question (Chenyang Lei, leichenyang7@gmail.com)
