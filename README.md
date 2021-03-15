# LSFNet (TMM)
Pytorch code for "**Low-light Image Restoration with Short- and Long-exposure Raw Pairs**" [[Paper]](https://arxiv.org/abs/2007.00199)

(Noting: The source code is a coarse version for reference and the model provided may not be optimal.)

## Prerequisites
* Python 3.6
* Pytorch 1.1
* CUDA 9.0
* rawpy 0.13.1

## Get Started
### Installation
The Deformable ConvNets V2 (DCNv2) module in our code adopts  [EDVR's implementation](https://github.com/xinntao/EDVR/tree/master/basicsr/models/ops).

You can compile the code according to your machine. 
```
cd ./dcn
python setup.py develop
```

Please make sure your machine has a GPU, which is required for the DCNv2 module.


### Train
1. Download the training dataset and use `gen_dataset.py` to package them in the h5py format.
2. Place the h5py file in `/Dataset/train/` or set the 'src_path' in `train.py` to your own path.
3. You can set any training parameters in `train.py`. After that, train the model:
```
cd $LSFNet_ROOT
python train.py
```

### Test
1. Download the trained models (uploading soon) and place them in `/ckpt/`.
2. Place the testing dataset in `/Dataset/test/` or set the testing path in `test.py` to your own path.
3. Set the parameters in `test.py` (eg. 'epoch_test', 'gray' and etc.)
3. test the trained models:
```
cd $LSFNet_ROOT
python test.py
```

## Citation
If you find the code helpful in your research or work, please cite the following papers.
```
@article{chang2021low,
  title={Low-light Image Restoration with Short-and Long-exposure Raw Pairs},
  author={Chang, Meng and Feng, Huajun and Xu, Zhihai and Li, Qi},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE}
}
```

## Acknowledgments
The DCNv2 module in our code adopts from [EDVR's implementation](https://github.com/xinntao/EDVR/tree/master/basicsr/models/ops).
