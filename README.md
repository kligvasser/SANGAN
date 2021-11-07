# Sparsity Aware Normalization for GANs
An official PyTorch implentation of SANGAN.

<p align="center">
  <img width="1100" height="549" src="/figures/figure.png">
</p>

Please refer our [paper](https://www.aaai.org/AAAI21Papers/AAAI-2034.KligvasserI.pdf) for more details.


## Citation
If you use this code for your research, please cite our work:

```
@inproceedings{kligvasser2021sparsity,
  title={Sparsity Aware Normalization for GANs},
  author={Kligvasser, Idan and Michaeli, Tomer},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={9},
  pages={8181--8190},
  year={2021}
}
```

## Code

### Clone repository

Clone this repository into any place you want.

```
git clone https://github.com/kligvasser/SANGAN
cd SANGAN
```

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code requires PyTorch 1.7+ and python 3+.

### Super-resoltution
Pretrained models are avaible at: [LINK](https://www.dropbox.com/s/gs4jwpel9cz463z/pre_trained.zip?dl=0).


#### Dataset preparation
For the super-resolution task, the dataset should contains a low and high resolution pairs, in folder structure of:

```txt
train
├── img
├── img_x2
├── img_x4
val
├── img
├── img_x2
├── img_x4
```

You may prepare your own data by using the matlab script:

```
./super-resolution/scripts/matlab/bicubic_subsample.m
```

Or download a prepared dataset based on the [BSD](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) datasets from [LINK](https://www.dropbox.com/s/o1nzpr9q7vup8b7/bsdvoc.zip?dl=0).

#### Train SRGAN x4 PSNR model
```
python3 main.py --root <path-to-dataset> --gen-model g_srgan --gen-model-config "{'scale':4}" --scale 4 --reconstruction-weight 1.0 --perceptual-weight 0 --adversarial-weight 0 --crop-size 40
```

#### Train SAN-SRGAN x4 model
```
python3 main.py --root <path-to-dataset> --dis-betas 0.5 0.9 --gen-model g_srgan --dis-model d_sanvanilla --dis-model-config "{'max_features':512, 'gain':1.05}" --scale 4 --reconstruction-weight 1 --perceptual-weight 1 --adversarial-weight 0.1 --crop-size 40 --gen-to-load <path-to-psnr-pretrained-pt> --results-dir ./results/san-srgan/
```

#### Eval SAN-SRGAN x4 model
```
python3 main.py --root <path-to-dataset> --gen-model g_srgan --gen-model-config "{'scale':4}" --scale 4 --evaluation --gen-to-load <path-to-pretrained-pt>
```
