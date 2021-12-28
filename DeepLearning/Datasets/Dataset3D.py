import os
import glob
import torch
import numpy as np
import nibabel as nib
import torchio as tio
import torch.nn.functional as F

from torch.utils.data import Dataset


class VolumetricDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 label_path: str,
                 num_classes: int,
                 slices_each_sub_volume: int,
                 transform: tio.Transform = None):
        imgs = sorted(glob.glob(os.path.join(data_path, '*.nii.gz')))
        labels = sorted(glob.glob(os.path.join(label_path, '*.nii.gz')))

        self.num_classes = num_classes
        assert len(imgs) == len(labels)
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))
        self.data = list()
        self.label_info = dict()
        self.transform_fn = transform
        for img, label in zip(imgs, labels):
            label_obj = nib.load(label)
            label_affine = label_obj.affine
            label_data = label_obj.get_fdata()
            label_data = np.transpose(label_data, (2, 0, 1))
            # for visualising the model predictions
            num_slices = label_data.shape[0]
            self.label_info[label] = (num_slices, label_affine)
            # pad the volume with empty slices
            img_data = torch.squeeze(rescale(
                tio.Subject(image=tio.ScalarImage(img))
            )["image"].data, dim=0).numpy()
            img_data = np.transpose(img_data, (2, 0, 1))
            num_slices = label_data.shape[0]
            # figure out the number of sub-volumes this MRI scan is about to be broken down to
            num_sub_volumes = num_slices // slices_each_sub_volume
            for i in range(num_sub_volumes):
                start_idx = i * slices_each_sub_volume
                end_dix = (i + 1) * slices_each_sub_volume
                img_data_this = img_data[start_idx:end_dix]
                label_data_this = label_data[start_idx:end_dix]
                self.data.append((img_data_this, label_data_this, img, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data, label_data, img, label = self.data[idx]
        img_data, label_data = torch.from_numpy(img_data), torch.from_numpy(label_data).to(torch.int64)
        if self.num_classes >= 2:
            label_data = F.one_hot(label_data, num_classes=self.num_classes)
            label_data = label_data.permute((3, 0, 1, 2))
        img_data, label_data = torch.unsqueeze(img_data, dim=0), label_data.to(torch.float32)
        return img_data, label_data
