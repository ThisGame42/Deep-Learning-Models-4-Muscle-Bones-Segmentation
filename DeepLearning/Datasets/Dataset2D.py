import os
import glob
import torch
import numpy as np
import torchio as tio
import nibabel as nib
import torch.nn.functional as F

from torch.utils.data import Dataset


class SliceDataset(Dataset):
    """
        A dataset class that processes 3D scans slice-wise. 3D data, such as MRI scans, are broken down
        to slices and stored in the object of this class. This means the object of this class returns
        pairs of 2D slices when been called.

        Usage:

        training_set = SliceDataset("./path/to/imgs", "./path/to/labels")

        data_loader = torch.utils.data.DataLoader(dataset=training_set, ...)

        ...

        for batch in data_loader():
            x, y = batch

            ...
    """

    def __init__(self,
                 data_path: str,
                 label_path: str,
                 num_classes: int,
                 transform: tio.Transform = None):
        """
        Initialise this class. Note that data_path and label_path assumed to be folders where Nifti images and their
        segmentation labels (both in .nii.gz format) are stored. This class also assumes that the number of Nifti images
        and the number of segmentation labels are equal.
        :param data_path: Path to the folder where input images are stored
        :param label_path: Path to the folder where segmentation labels are stored
        :param num_classes: Number of classes in the reference labels
        """
        imgs = sorted(glob.glob(os.path.join(data_path, '*.nii.gz')))
        labels = sorted(glob.glob(os.path.join(label_path, '*.nii.gz')))

        assert len(imgs) == len(labels)
        rescale = tio.RescaleIntensity(out_min_max=(0, 1))

        self.data = list()
        self.num_classes = num_classes
        self.transform_fn = transform
        for img, label in zip(imgs, labels):
            label_data = nib.load(label).get_fdata().astype(np.uint8)
            img_data = tio.Subject(image=tio.ScalarImage(img))
            img_data = torch.squeeze(rescale(img_data)["image"].data, dim=0).numpy().astype(np.float32)
            assert img_data.shape == label_data.shape
            # the shape of the image/label assumed to be
            # width x height x depth (depth is also known as num of slices)
            for slice_num in range(img_data.shape[2]):
                self.data.append((img_data[..., slice_num],
                                  label_data[..., slice_num]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        x, y = torch.from_numpy(data[0]).to(torch.float64), torch.from_numpy(data[1]).to(torch.int64)
        x, y = torch.unsqueeze(x, dim=0), y
        if self.num_classes >= 2:
            y = F.one_hot(y, num_classes=self.num_classes)
            y = y.permute(2, 0, 1)
        return x, y.to(torch.float32)
