import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from torch.utils.data import DataLoader

from Utils.Device import get_device
from Utils.Visualisation import plot_loss

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def train_hybrid3D(model_3d: nn.Module,
                   model_2d: nn.Module,
                   optimiser: optim.Optimizer,
                   loss_fn: nn.Module,
                   lr_scheduler: optim.lr_scheduler,
                   training_set_loader: DataLoader,
                   val_set_loader: DataLoader,
                   num_epochs: int,
                   output_path: str) -> None:
    model_3d = model_3d.to(get_device())

    model_2d = model_2d.to(get_device())
    model_2d.eval()
    for p in model_2d.parameters():
        p.requires_grad = False

    avg_training_loss, avg_val_loss = list(), list()
    for epoch in range(num_epochs):
        model_3d.train()
        epoch_t_loss, epoch_v_loss, idx = 0., 0., 0
        for idx, batch in enumerate(training_set_loader):
            optimiser.zero_grad()
            input_images = batch[0].to(get_device(), dtype=torch.float32)
            input_2d = torch.unsqueeze(input_images[0, 0, ...], dim=1)
            logits_2d, feature_2d = model_2d(input_2d)
            output_2d = F.softmax(logits_2d, dim=1)
            output_2d = output_2d.permute(1, 0, 2, 3)
            feature_2d = feature_2d.permute(1, 0, 2, 3)
            output_2d = torch.unsqueeze(output_2d, dim=0)
            feature_2d = torch.unsqueeze(feature_2d, dim=0)
            input_3d = torch.cat([output_2d, input_images], dim=1)
            pred_labels = model_3d(input_3d, feature_2d)
            ref_labels = batch[1].to(get_device())
            loss_val = loss_fn(pred_labels, ref_labels)
            loss_val.backward()
            optimiser.step()
            epoch_t_loss += loss_val.item()

        avg_training_loss.append(epoch_t_loss / (idx + 1))
        print(f"The average training loss at epoch: {epoch + 1} was {epoch_t_loss / (idx + 1)}.")

        with torch.no_grad():
            model_3d.eval()
            for idx, batch in enumerate(val_set_loader):
                input_images = batch[0].to(get_device(), dtype=torch.float32)
                input_2d = torch.unsqueeze(input_images[0, 0, ...], dim=1)
                logits_2d, feature_2d = model_2d(input_2d)
                output_2d = F.softmax(logits_2d, dim=1)
                output_2d = output_2d.permute(1, 0, 2, 3)
                feature_2d = feature_2d.permute(1, 0, 2, 3)
                output_2d = torch.unsqueeze(output_2d, dim=0)
                feature_2d = torch.unsqueeze(feature_2d, dim=0)
                input_3d = torch.cat([output_2d, input_images], dim=1)
                pred_labels = model_3d(input_3d, feature_2d)
                ref_labels = batch[1].to(get_device())
                loss_val = loss_fn(pred_labels, ref_labels)
                epoch_v_loss += loss_val.item()

        epoch_val_loss = epoch_v_loss / (idx + 1)
        lr_scheduler.step(epoch_val_loss)
        avg_val_loss.append(epoch_val_loss)
        print(f"The average validation loss at epoch: {epoch + 1} was {epoch_val_loss}.")

    model_name = type(model_3d).__name__
    time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    torch.save(model_3d.state_dict(), os.path.join(output_path, f"{model_name}_{time_stamp}.pth"))
    print(f"Model weights saved to {output_path} with the name: {model_name}_{time_stamp}.pth")

    plot_loss(training_loss_=avg_training_loss,
              val_loss_=avg_val_loss,
              num_epochs=num_epochs,
              path_dice_plot=os.path.join(output_path, f"{model_name}_{time_stamp}.png"))
    print(f"The loss graph was saved to {output_path} with the name: {model_name}_{time_stamp}.png")


def train_hybrid2D(model: nn.Module,
                   optimiser: optim.Optimizer,
                   loss_fn: nn.Module,
                   lr_scheduler: optim.lr_scheduler,
                   training_set_loader: DataLoader,
                   val_set_loader: DataLoader,
                   num_epochs: int,
                   output_path: str) -> None:
    model = model.to(get_device())

    avg_training_loss, avg_val_loss = list(), list()
    for epoch in range(num_epochs):
        model.train()
        epoch_t_loss, epoch_v_loss, idx = 0., 0., 0
        for idx, batch in enumerate(training_set_loader):
            optimiser.zero_grad()
            input_images = batch[0].to(get_device(), dtype=torch.float32).float()
            pred_labels = model(input_images)[0]
            ref_labels = batch[1].to(get_device())
            loss_val = loss_fn(pred_labels, ref_labels)
            loss_val.backward()
            optimiser.step()
            epoch_t_loss += loss_val.item()

        avg_training_loss.append(epoch_t_loss / (idx + 1))
        print(f"The average training loss at epoch: {epoch + 1} was {epoch_t_loss / (idx + 1)}.")

        with torch.no_grad():
            model.eval()
            for idx, batch in enumerate(val_set_loader):
                input_images = batch[0].to(get_device(), dtype=torch.float32).float()
                pred_labels = model(input_images)[0]
                ref_labels = batch[1].to(get_device())
                loss_val = loss_fn(pred_labels, ref_labels)
                epoch_v_loss += loss_val.item()

        epoch_val_loss = epoch_v_loss / (idx + 1)
        lr_scheduler.step(epoch_val_loss)
        avg_val_loss.append(epoch_val_loss)
        print(f"The average validation loss at epoch: {epoch + 1} was {epoch_val_loss}.")

    save_model_plots(model, output_path, avg_training_loss,
                     avg_val_loss, num_epochs)


def train_model(model: nn.Module,
                optimiser: optim.Optimizer,
                loss_fn: nn.Module,
                lr_scheduler: optim.lr_scheduler,
                training_set_loader: DataLoader,
                val_set_loader: DataLoader,
                num_epochs: int,
                output_path: str) -> None:
    model = model.to(get_device())

    avg_training_loss, avg_val_loss = list(), list()
    for epoch in range(num_epochs):
        model.train()
        epoch_t_loss, epoch_v_loss, idx = 0., 0., 0
        for idx, batch in enumerate(training_set_loader):
            optimiser.zero_grad()
            input_images = batch[0].to(get_device(), dtype=torch.float32)
            pred_labels = model(input_images)
            ref_labels = batch[1].to(get_device())
            loss_val = loss_fn(pred_labels["seg_results"], ref_labels)
            loss_val.backward()
            optimiser.step()
            del pred_labels, input_images, ref_labels
            epoch_t_loss += loss_val.item()

        avg_training_loss.append(epoch_t_loss / (idx + 1))
        print(f"The average training loss at epoch: {epoch + 1} was {epoch_t_loss / (idx + 1)}.")

        with torch.no_grad():
            model.eval()
            for idx, batch in enumerate(val_set_loader):
                input_images = batch[0].to(get_device(), dtype=torch.float32)
                pred_labels = model(input_images)
                ref_labels = batch[1].to(get_device())
                loss_val = loss_fn(pred_labels["seg_results"], ref_labels)
                epoch_v_loss += loss_val.item()

        epoch_val_loss = epoch_v_loss / (idx + 1)
        lr_scheduler.step(epoch_val_loss)
        avg_val_loss.append(epoch_val_loss)
        print(f"The average validation loss at epoch: {epoch + 1} was {epoch_val_loss}.")

    save_model_plots(model, output_path, avg_training_loss,
                     avg_val_loss, num_epochs)


def save_model_plots(model, output_path, avg_training_loss,
                     avg_val_loss, num_epochs):
    model_name = type(model).__name__
    time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    torch.save(model.state_dict(), os.path.join(output_path, f"{model_name}_{time_stamp}.pth"))
    print(f"Model weights saved to {output_path} with the name: {model_name}_{time_stamp}.pth")

    plot_loss(training_loss_=avg_training_loss,
              val_loss_=avg_val_loss,
              num_epochs=num_epochs,
              path_dice_plot=os.path.join(output_path, f"{model_name}_{time_stamp}.png"))
    print(f"The loss graph was saved to {output_path} with the name: {model_name}_{time_stamp}.png")
