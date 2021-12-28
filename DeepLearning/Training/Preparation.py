import os
import sys
import torch
import random
import numpy as np
import torch.optim as optim

import DeepLearning.Models.UNet as UNet
import DeepLearning.Models.UNet3D as UNet3D
import DeepLearning.LossFunctions.DiceLoss as DiceFn
import DeepLearning.Models.H_DenseUNet as H_DenseUNet
import DeepLearning.LossFunctions.CrossEntropy as CSEFn
import DeepLearning.LossFunctions.DiceCSELoss as DiceCSE

from torch.utils.data import DataLoader
from DeepLearning.LossFunctions.FocalLoss import FocalLoss
from DeepLearning.LossFunctions.TverskyLoss import TverskyLoss
from DeepLearning.Datasets.Dataset2D import SliceDataset
from DeepLearning.Datasets.Dataset3D import VolumetricDataset
from DeepLearning.Testing.Test import test_model
from DeepLearning.Training.Train import train_model, train_hybrid2D, train_hybrid3D

RANDOM_SEED = 42


def worker_init_fn(worker_id):
    random.seed(RANDOM_SEED + worker_id)
    torch.manual_seed(RANDOM_SEED + worker_id)
    np.random.seed(RANDOM_SEED + worker_id)


def start_testing(args: sys.argv):
    all_num_slices = 96
    testing_imgs_path = os.path.join(args.testing_path, "imgs")
    testing_labels_path = os.path.join(args.testing_path, "labels")
    num_classes = int(args.num_classes)
    is_2d = bool(args.is_2d)
    num_slices = int(args.num_slices)
    model_name = str(args.model_name)
    batch_size = int(args.batch_size)
    test_dataset = get_dataset(num_slices, is_2d, testing_imgs_path, testing_labels_path,
                               num_classes, None)
    collection_point = all_num_slices // num_slices
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             pin_memory=True, num_workers=4, shuffle=False, worker_init_fn=worker_init_fn)
    model = load_trained_model(model_name,
                               num_classes,
                               args.model_path)
    test_model(model, test_loader, collection_point)


def start_training(args: sys.argv):
    training_img_path: str = args.training_img_path
    training_label_path: str = args.training_label_path
    val_img_path: str = args.val_img_path
    val_label_path: str = args.val_label_path
    lossfn_name: str = args.loss_fn
    optimiser_name: str = args.optimiser
    num_epochs = int(args.num_epochs)
    output_path: str = args.output_path
    batch_size = int(args.batch_size)
    num_classes = int(args.num_classes)
    lr = float(args.learning_rate)
    model_name: str = args.model_name
    pre_trained = False if args.pre_training is None else args.pre_training
    is_2d: bool = args.is_2d
    num_slices = int(args.num_slices)
    if not args.fine_tuning:
        model = load_new_model(model_name,
                               num_classes,
                               pre_trained)
    else:
        model = load_trained_model(model_name,
                                   num_classes,
                                   args.model_path)

    optimiser = get_optimiser(optimiser_name, model, lr)
    loss_fn = get_lossfn(lossfn_name, num_classes)

    training_set = get_dataset(num_slices, is_2d, training_img_path, training_label_path, num_classes,
                               transform_fn=None)
    val_set = get_dataset(num_slices, is_2d, val_img_path, val_label_path, num_classes,
                          transform_fn=None)
    training_loader = DataLoader(dataset=training_set, batch_size=batch_size,
                                 pin_memory=True, num_workers=4, shuffle=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size,
                            pin_memory=True, num_workers=4, shuffle=True, worker_init_fn=worker_init_fn)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimiser, mode="min", factor=0.5,
                                                        patience=num_epochs // 10, verbose=True)

    if "h_denseunet" not in model_name.lower():
        train_model(model=model,
                    optimiser=optimiser,
                    loss_fn=loss_fn,
                    lr_scheduler=lr_scheduler,
                    training_set_loader=training_loader,
                    val_set_loader=val_loader,
                    num_epochs=num_epochs,
                    output_path=output_path)
    # TODO
    # test this

    # else:
    #     print("Training hybrid.")
    #     if "3D" not in model_name:
    #         train_hybrid2D(model=model,
    #                        optimiser=optimiser,
    #                        loss_fn=loss_fn,
    #                        lr_scheduler=lr_scheduler,
    #                        training_set_loader=training_loader,
    #                        val_set_loader=val_loader,
    #                        num_epochs=num_epochs,
    #                        output_path=output_path)
    #     else:
    #         train_hybrid3D(model_3d=model,
    #                        model_2d=model_2d,
    #                        optimiser=optimiser,
    #                        loss_fn=loss_fn,
    #                        lr_scheduler=lr_scheduler,
    #                        training_set_loader=training_loader,
    #                        val_set_loader=val_loader,
    #                        num_epochs=num_epochs,
    #                        output_path=output_path)


def get_dataset(num_slices, is_2d, img_path, label_path, num_classes, transform_fn):
    if is_2d:
        dataset = SliceDataset(data_path=img_path,
                               label_path=label_path,
                               num_classes=num_classes,
                               transform=transform_fn)
    else:
        dataset = VolumetricDataset(data_path=img_path,
                                    label_path=label_path,
                                    num_classes=num_classes,
                                    slices_each_sub_volume=num_slices,
                                    transform=transform_fn)
    return dataset


def load_trained_model(model_name, num_classes, model_path):
    trained_model = load_new_model(model_name, num_classes, False)
    trained_model.load_state_dict(torch.load(model_path))
    return trained_model


def load_new_model(model_name, num_classes, pre_trained):
    if model_name.lower() == "unet":
        model = UNet.UNet(in_channels=1,
                          classes=num_classes,
                          base_filter=64)
    elif model_name.lower() == "unet3d":
        model = UNet3D.UNet3D(in_channels=1,
                              num_classes=num_classes,
                              base_filter=64)
    elif "h_dense" in model_name.lower():
        if "2d" in model_name.lower():
            model = H_DenseUNet.DenseUNet2D(num_classes=num_classes,
                                            pretrained=pre_trained)
        else:
            model = H_DenseUNet.DenseUNet3D(num_classes=num_classes)
    else:
        raise ValueError(f"{model_name} not recognised.")
    return model


def get_optimiser(optimiser_name, model, lr):
    if "adam" in optimiser_name.lower():
        if "w" not in optimiser_name.lower():
            optimiser = optim.Adam(params=model.parameters(), lr=lr)
        else:
            optimiser = optim.AdamW(params=model.parameters(), lr=lr)
    else:
        raise ValueError(f"{optimiser_name} not supported at the moment.")
    return optimiser


def get_lossfn(lossfn_name, num_classes):
    if "bce_dice" in lossfn_name.lower():
        loss_fn = DiceCSE.DiceBCELoss()
    elif "dice" in lossfn_name.lower():
        loss_fn = DiceFn.WeightedDiceLoss(num_classes=num_classes, ignore_background=False)
    elif "cse" in lossfn_name.lower():
        loss_fn = CSEFn.SoftCrossEntropy()
    elif "focal" in lossfn_name.lower():
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    elif "tversky" in lossfn_name.lower():
        loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
    else:
        raise ValueError(f"{lossfn_name} not supported.")
    return loss_fn
