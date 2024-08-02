NN
# AttU_Net Training

This repository contains the implementation and training script for "Enhancing Polyp Segmentation in Colorectal
Cancer using Attention U-Net Model", which is used for image segmentation tasks.



## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction
According to the Canadian Cancer Society, colorectal cancer is the second leading cause of cancer death in men and the third in women.[2] The incidence of colorectal cancer in adults under 50 is increasing, particularly among those aged 20-29 and 30-39.[1] Colon polyps, clusters of cells forming on the colon lining, are generally benign but can develop into colorectal cancer.[3] Colonoscopy is the most effective method for capturing medical images to detect colon polyps. Analyzing these images using deep learning techniques for polyp segmentation can significantly improve diagnosis by quickly identifying abnormalities in the colon, allowing for earlier intervention and reducing the risk of colorectal cancer progression.
Recent studies have highlighted the potential of various deep-learning architectures in medical image analysis. Among these, attention mechanisms in deep learning models have shown great promise, especially in image segmentation. This research aims to leverage Attention U-Net architecture's capabilities to improve polyps' segmentation accuracy in colonoscopy images. 





## Model Architecture

The model architecture includes different variations:
- U_Net
- R2U_Net
- AttU_Net


## Dependencies
- Python 3.9
- argparse
- PyTorch
- Optuna (if hyperparameter tuning is enabled)
- Other dependencies as required by your project

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Dataset

Download  kvasir-dataset
https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset/data


## Training the Model
We have used J. Lee's "Image Segmentation" GitHub repository (https://github.com/LeeJunHyun/Image_Segmentation) as a baseline, and the Ksavir dataset has been implemented on top of it.

To run the script, use the following command:

```bash
python main_v2_hyper_add.py \
--image_size 224 \
--t 3 \
--img_ch 3 \
--output_ch 1 \
--num_epochs 2 \
--num_epochs_decay 70 \
--batch_size 2 \
--num_workers 4 \
--lr 0.0002 \
--beta1 0.5 \
--beta2 0.999 \
--augmentation_prob 0.4 \
--decay_ratio 0.5 \
--log_step 2 \
--val_step 2 \
--mode train \
--model_type AttU_Net \
--model_path ../hyperOutput/models \
--test_model_path <MODEL PATH FOR EVALUATION > \
--image_path <IMAGE PATH FOR EVALUATION > \
--result_path ../hyperOutput/result/ \
--image_path_dir ../data/kvasir-dataset/kvasir-seg/Kvasir-SEG/ \
--cuda_idx 1 

```
Options
- image_size: Size of the input image. Default is 224.
- t: Recurrent step for R2U_Net or R2AttU_Net. Default is 3.
- img_ch: Number of image channels. Default is 3.
- output_ch: Number of output channels. Default is 1.
- num_epochs: Number of training epochs. Default is 2.
- num_epochs_decay: Number of epochs after which learning rate decay is applied. Default is 70.
- batch_size: Batch size for training. Default is 2.
- num_workers: Number of workers for data loading. Default is 4.
- lr: Learning rate. Default is 0.0002.
- beta1: Momentum1 in Adam optimizer. Default is 0.5.
- beta2: Momentum2 in Adam optimizer. Default is 0.999.
- augmentation_prob: Probability of applying data augmentation. Default is 0.4.
- decay_ratio: Ratio of learning rate decay. Default is 0.5.
- log_step: Frequency of logging training information. Default is 2.
- val_step: Frequency of validation. Default is 2.
- mode: Mode of operation (train or test). Default is 'train'.
- model_type: Type of model to use ('U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net'). Default is 'AttU_Net'.
- model_path: Path to save the trained model. Default is ../hyperOutput/models.
- test_model_path: Path to a pretrained model for testing. 
- image_path: Path to a specific image for testing. 
- result_path: Path to save the results. Default is ../hyperOutput/result/.
- image_path_dir: Directory containing images for training/testing. Default is ../data/kvasir-dataset/kvasir-seg/Kvasir-SEG/.
- cuda_idx: Index of the CUDA device to use. Default is 1.
- tune: Flag to enable hyperparameter tuning with Optuna. Default is True.


## EVALUATION 


```bash
python main_v2_hyper_add.py \
--image_size 224 \
--t 3 \
--img_ch 3 \
--output_ch 1 \
--num_epochs 2 \
--num_epochs_decay 70 \
--batch_size 2 \
--num_workers 4 \
--lr 0.0002 \
--beta1 0.5 \
--beta2 0.999 \
--augmentation_prob 0.4 \
--decay_ratio 0.5 \
--log_step 2 \
--val_step 2 \
--mode eval_test \
--model_type AttU_Net \
--model_path ../hyperOutput/models \
--test_model_path <MODEL PATH FOR EVALUATION > \
--image_path <IMAGE PATH FOR EVALUATION > \
--result_path ../hyperOutput/result/ \
--image_path_dir ../data/kvasir-dataset/kvasir-seg/Kvasir-SEG/ \
--cuda_idx 1 

```

EVALUATION OF A IMAGE DEPEND ON >> image_path

```bash
python main_v2_hyper_add.py \
--image_size 224 \
--t 3 \
--img_ch 3 \
--output_ch 1 \
--num_epochs 2 \
--num_epochs_decay 70 \
--batch_size 2 \
--num_workers 4 \
--lr 0.0002 \
--beta1 0.5 \
--beta2 0.999 \
--augmentation_prob 0.4 \
--decay_ratio 0.5 \
--log_step 2 \
--val_step 2 \
--mode indivi_test \
--model_type AttU_Net \
--model_path ../hyperOutput/models \
--test_model_path <MODEL PATH FOR EVALUATION > \
--image_path <IMAGE PATH FOR EVALUATION > \
--result_path ../hyperOutput/result/ \
--image_path_dir ../data/kvasir-dataset/kvasir-seg/Kvasir-SEG/ \
--cuda_idx 1 

```

## HISTORY
The final run and output is found in (https://github.com/MatthewLamBok/Assignment_NN/blob/main/main_v3_google_colab.ipynb) 


## Acknowledgements

[1] D. E. O’Sullivan, R. J. Hilsden, Y. Ruan, N. Forbes, S. J. Heitman, and D. R. Brenner, “The incidence of young-onset colorectal cancer in
Canada continues to increase,” Cancer Epidemiol., vol. 69, p. 101828, Dec. 2020, doi: 10.1016/j.canep.2020.101828.
[2] C. C. S. / S. canadienne du cancer, “Colorectal cancer statistics,” Canadian Cancer Society. Accessed: Jun. 23, 2024. [Online]. Available: https://cancer.ca/en/cancer-information/cancer-types/colorectal/statistics
[3] M. Meseeha and M. Attia, “Colon Polyps,” in StatPearls, Treasure Island (FL): StatPearls Publishing, 2024. Accessed: Jun. 23, 2024. [Online]. Available: http://www.ncbi.nlm.nih.gov/books/NBK430761/
