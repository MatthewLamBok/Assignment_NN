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

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Dataset

## CODE
We have used J. Lee's "Image Segmentation" GitHub repository (https://github.com/LeeJunHyun/Image_Segmentation) as a baseline, and the Ksavir dataset has been implemented on top of it.

## References

To install the required dependencies, run:
