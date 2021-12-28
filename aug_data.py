import os
import glob
import torch
import numpy as np
import nibabel as nib
import torchio as tio
import torch.nn.functional as F

from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance


def pad_volume(slices_pad: int,
               data: np.ndarray):
    up_bound_old = data.shape[2]
    up_bound_new = slices_pad
    new_data = np.zeros((data.shape[0], data.shape[1], up_bound_new), dtype=np.float32)

    if up_bound_old <= up_bound_new:
        new_data[..., 0:up_bound_old] = data
    else:
        new_data = data[..., 0:up_bound_new]
    return new_data


def load_data(root_path):
    path_img = os.path.join(root_path, "images")
    path_label = os.path.join(root_path, "labels")

    images = sorted(glob.glob(os.path.join(path_img, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(path_label, "*.nii.gz")))

    return np.array(images), np.array(labels)


def get_affine(index):
    if index == 0:
        return tio.RandomAffine(scales=(0.9, 1.1))
    elif index == 1:
        return tio.RandomAffine(degrees=5)
    elif index == 2:
        return tio.RandomAffine(translation=5)
    else:
        return tio.Compose([
            tio.RandomAffine(scales=(0.9, 1.1)),
            tio.RandomAffine(degrees=2),
            tio.RandomAffine(translation=2)
        ])


def augment_data(root_path, output_path):
    images, labels = load_data(root_path)
    num_augmentation_iter = 4

    for i in range(num_augmentation_iter):
        t_fn = get_affine(i)
        for img, label in zip(images, labels):
            if i == 0:
                print(f"Processing {img}.")
            new_img_name = os.path.join(output_path, "images",
                                        os.path.basename(img.replace(".nii.gz", f"_{i}.nii.gz")))
            new_label_name = os.path.join(output_path, "labels",
                                          os.path.basename(label.replace(".nii.gz", f"_{i}.nii.gz")))
            if len(nib.load(img).get_fdata().shape) == 4:
                img_data = tio.ScalarImage(
                    tensor=np.transpose(nib.load(img).get_fdata(), (3, 0, 1, 2))
                )
            else:
                img_data = tio.ScalarImage(img)
            label_data = tio.LabelMap(label)
            subject = tio.Subject(image=img_data,
                                  label=label_data)
            ret_data = t_fn(subject)
            img_data, img_affine = torch.squeeze(ret_data["image"].data, dim=0).numpy(), \
                                   ret_data["image"].affine
            img_data = np.flip(np.transpose(img_data, (1, 2, 3, 0)), axis=0) if len(img_data.shape) == 4 else img_data
            label_data, label_affine = torch.squeeze(ret_data["label"].data, dim=0).numpy(), \
                                       ret_data["label"].affine

            assert np.all(np.equal(np.unique(label_data), list(range(14))))
            nib.save(nib.Nifti1Image(img_data, img_affine), new_img_name)
            nib.save(nib.Nifti1Image(label_data, label_affine), new_label_name)


def pad_data(root_path):
    pad_slices = 96
    images, labels = load_data(root_path)
    img_list, label_list = dict(), dict()
    for img, label in zip(images, labels):
        real_img = nib.load(img)
        real_label = nib.load(label)
        img_list[img] = (real_img.get_fdata(), real_img.affine)
        label_list[label] = (real_label.get_fdata(), real_label.affine)

    for img, label in zip(images, labels):
        print(f"Processing {img}, {label}")
        old_img, old_affine = img_list[img]
        old_label, old_affine_label = label_list[label]
        img_data, label_data = pad_volume(pad_slices, old_img), pad_volume(pad_slices, old_label)
        nib.save(nib.Nifti1Image(img_data, old_affine), img)
        nib.save(nib.Nifti1Image(label_data, old_affine_label), label)


def load_from_f(file_path: str):
    data = list()
    with open(file_path, "r") as f:
        for line in f:
            data.append(line.rstrip())
    return sorted(data)


def write_f(file_name: str, data: list):
    with open(file_name, "w+") as f:
        print(*data, sep="\n", file=f)


if __name__ == "__main__":
    pad_data("") # replace with path or your data
    augment_data("path_to_your_data", "path_to_your_output_folder")
