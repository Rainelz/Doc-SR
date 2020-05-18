This repo is based on the [work](https://github.com/open-mmlab/mmsr) done by open--mmlab. 

This fork is an adaptation of ESRGAN focused on Super Resolving and cleaning grayscale document images in order to improve OCR results by using a synthetic image generator that provides data with different layouts and type of noises (on and off), thus augmentations have been left out from this repo.

# My edits
- efficient-net both as discriminator and as perceptual network.

    The perceptual features are extracted from multiple blocks of the network as well as from the last feature block before 
    activation as suggested in [ESRGAN](https://arxiv.org/abs/1809.00219). 

- Use an edge loss to improve the character reconstruction by using a convolution with a LoG kernel. 

- Add label smoothing and flooded loss on D

- Random rotation on samples (use carefully, interpolations worsen the GT)

- Add a mixed upsample block (not used)

- Minors on logging and paths
# 3-steps training
To get better results, I trained the net in multiple steps:
1. From LR to HR using non spoiled images (from scratch)
2. From LR_spoiled to HR using spoiled images and transfering from step 1
3. From a given domain dataset to HR with GAN enabled, allowing G to finetune its weights and try to make the domain dataset look similar to the generated one (transferring from step 2)
# Results
Even though the project was focused on optimizing a custom dataset with different sizes, a restricted font set, and different noises, here are some 2x results on images from rvl-cdip using a model from step 2
- [1](https://imgsli.com/MTY1MTQ)
- [2](https://imgsli.com/MTY1MjA)
- [3](https://imgsli.com/MTY1MjI)
- [4](https://imgsli.com/MTY1MjM)
- [White-on-black text](https://imgsli.com/MTY1MjE)
- [5](https://imgsli.com/MTY1MTU)
- [6](https://imgsli.com/MTY1MTY)
- [7](https://imgsli.com/MTY1MTc)
- [8](https://imgsli.com/MTY1MTk)

[TODO]
- Test robust-loss
- Add finetuning yml
- Add+test attention based arch

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
