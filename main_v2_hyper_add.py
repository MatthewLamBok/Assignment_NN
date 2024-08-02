import itertools
import os
import random
import re
from glob import glob
import matplotlib.pyplot as plt
import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split
from random import sample
from PIL import Image
import pandas as pd
import optuna
import argparse
import os
from Image_Segmentation.solver import Solver
from Image_Segmentation.data_loader import get_loader
from torch.backends import cudnn
import random

def display_image_and_mask(image_path, mask_path):
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.title('Image and Mask')
    plt.axis('off')

    plt.show()


def dataset_init(image_path_dir, display_bool= False):
    num_dataset = 1

    image_path_dataset_1= image_path_dir + 'images'
    mask_path_dataset_1= image_path_dir + 'masks'

    print(len(os.listdir(image_path_dataset_1)), len(os.listdir(mask_path_dataset_1)))


    if display_bool== True:
        sample_image_1 = os.path.join(image_path_dataset_1, os.listdir(image_path_dataset_1)[0])
        sample_mask_1 = os.path.join(mask_path_dataset_1, os.listdir(mask_path_dataset_1)[0])
        display_image_and_mask(sample_image_1, sample_mask_1)



    image_paths_1 = [os.path.join(image_path_dataset_1, img) for img in os.listdir(image_path_dataset_1)]
    mask_paths_1 = [os.path.join(mask_path_dataset_1, img) for img in os.listdir(mask_path_dataset_1)]

 

    #YOU NEED TO REMOVE test with smaller dataset
    #length = 100
    #image_paths_1 = image_paths_1[ :length]
    #mask_paths_1 = mask_paths_1[ :length]
    #print(length)


    image_paths = image_paths_1
    mask_paths = mask_paths_1

    image_paths.sort()
    mask_paths.sort()


    paired_paths = list(zip(image_paths, mask_paths))



    return paired_paths

def objective(trial, config):
    paired_paths = dataset_init(config.image_path_dir)
    train_val_paths, test_paths = train_test_split(paired_paths, test_size=0.2, random_state=42)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=0.25, random_state=42)

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    if trial is not None:
        lr = trial.suggest_float('lr', 0.0000005, 0.0005)
        augmentation_prob = trial.suggest_float('augmentation_prob', 0.0, 0.7)
        epoch = trial.suggest_categorical('epoch', [100]) #100, 150, 200, 250
        decay_ratio = trial.suggest_float('decay_ratio', 0.0, 0.8)
    else:
        lr = config.lr
        augmentation_prob = config.augmentation_prob
        epoch = config.num_epochs
        decay_ratio = config.decay_ratio

    decay_epoch = int(epoch * decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)

    if config.mode == 'indivi_test' or 'indivi_test_dataset' or 'eval_test':
        config.batch_size = 1

    train_loader = get_loader(image_mask_pairs=train_paths,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_mask_pairs=val_paths,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_mask_pairs=test_paths,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)
    
    # Train and sample the images
    if config.mode == 'train':
        solver = Solver(config, train_loader, valid_loader, test_loader)
        DC = solver.train()
    if config.mode == 'eval_test':
        solver = Solver(config, train_loader, valid_loader, test_loader, config.test_model_path)
        solver.eval_test(top_k= 4)
        DC = None
    if config.mode == 'indivi_test':
        solver = Solver(config,  model_path_eval = config.test_model_path)
        solver.eval_image(image_path = config.image_path)
        DC = None


    return DC

def main(config):
    if config.tune:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, config), n_trials=50)
        
        print('Number of finished trials: ', len(study.trials))
        print('Best trial:')
        trial = study.best_trial
        print('  Value: ', trial.value)
        print('  Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
    else:
        objective(None, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')


    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)
    parser.add_argument('--decay_ratio', type=float, default=0.5)  # Default decay ratio


    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='AttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='../hyperOutput/models')
    parser.add_argument('--test_model_path', type=str, default='/home/mlam/Documents/BME_NN_course/Output/models/R2AttU_Net-200-0.0005-137-0.6380.pkl')
    parser.add_argument('--image_path', type=str, default='/home/mlam/Documents/BME_NN_course/data/kvasir-dataset/kvasir-seg/Kvasir-SEG/images/ck2bxw18mmz1k0725litqq2mc.jpg')
    parser.add_argument('--result_path', type=str, default='../hyperOutput/result/')
    parser.add_argument('--image_path_dir', type=str, default='../data/kvasir-dataset/kvasir-seg/Kvasir-SEG/')
  
    parser.add_argument('--cuda_idx', type=int, default=1)
    parser.add_argument('--tune', action='store_true', default=True, help='Flag to enable hyperparameter tuning with Optuna')


    config = parser.parse_args()
    main(config)
