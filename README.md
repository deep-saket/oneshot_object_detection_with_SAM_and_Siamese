# One shot object detection using SAM and siamese network
An interesting demo by combining [Segment Anything](https://ai.facebook.com/research/publications/segment-anything/) of Meta and a VGG based Siamese Net

![Segment Anything Result](./outputs/owlvit_segment_anything_output.jpg)


## Highlight
- Image-conditioned detection g


## Catelog
- [x] Image-conditioned detection
- [x] Siamese network based classification

## Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

To run the siamese clasification model install tensorflow using `pip install tensorflow`

More details can be found in [installation segment anything](https://github.com/facebookresearch/segment-anything#installation)

## Run Demo

- download segment-anything checkpoint
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

- download VGG siamese classifier weights from 

- Run demo (image paths and output_dir before running)
```bash
python demo_one_shot_detection.py --query_image_path ./demo_images/qr18.jpg --image_path ./demo_images/db3103.jpg --output_dir ./outputs 
```

## Siamese Classification Model
- This model was trained by me, using a custom build dataset on K80 to serve as POC for an earlier work
- The backbone of the model is VGG and the dataset has 80 different classes and all classes have same number of images
- Further this model was evaluated on 500ms videos containing the objects and the accuracy was found to be 90.7%
- I have trained this model using tensorflow.

## Reference
Please give applause for [IDEA-Research](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/segment_anything)

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

```