This repo is a [fork](https://github.com/open-mmlab/mmsr) of the work done by open--mmlab which I deeply thank. 
For any detail you can follow the original [README](https://github.com/open-mmlab/mmsr/blob/master/README.md).

This fork is an adaptation of ESRGAN focused on Super Resolving and cleaning grayscale document images. 

It implements an efficient-net both as discriminator and as perceptual network.

The perceptual features are extracted from multiple blocks of the network as well as from the last feature block before 
activation as suggested in ESRGAN. 

Moreover, this work implements an edge loss to improve the character reconstruction by using a convolution with a LoG kernel. 

[TODO]
- Test robust-loss
# Inference Requirements
- NVIDIA GPU + [CUDA >= 9.0](https://developer.nvidia.com/cuda-downloads) (Recommended)
- Python 3
- Python packages: `pip install -r inference_requirements.txt`

# Requirements
- NVIDIA GPU + [CUDA >= 9.0](https://developer.nvidia.com/cuda-downloads)
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download))
- [PyTorch >= 1.1](https://pytorch.org)
- Python packages: `pip install -r requirements.txt`
- Efficient-net: `git clone https://github.com/Rainelz/EfficientNet-PyTorch && pip install EfficientNet-PyTorch`

## Dataset Preparation
`cd codes/data_scripts`
- `python generate_mod_LR_bic.py --datapath $IMAGE_FOLDER --out "$PROCESSED" --downscale "$SCALE"
` will generate the downscaled version of your images

- `python create_lmdb.py --data-path "$PROCESSED/LR/x$SCALE" --out "$PROCESSED/train/LRx'$SCALE'" --name $DATASET_NAME`
will generate the lmdb folder of your downscaled images

## Training and Testing
`cd codes`

- `python train.py -opt options/$YOUR_YAML.yml`
- `python test.py -opt options/$YOUR_TEST_YAML.yml`

## License
This project is released under the Apache 2.0 license.
