import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from Utils.Device import get_device
from torch.utils.data import DataLoader
from Utils.Evaluation import compute_DSC


def test_model(model: nn.Module,
               test_set_loader: DataLoader,
               collection_point: int) -> None:
    model = model.to(get_device())
    dsc_all = list()
    with torch.no_grad():
        model.eval()
        dsc_interim = list()
        index = 0
        for idx, batch in enumerate(test_set_loader):
            input_images = batch[0].to(get_device(), dtype=torch.float32)
            pred_labels = model(input_images)
            ref_labels = batch[1].to(get_device())
            dsc = compute_DSC(predictions=F.softmax(pred_labels, dim=1),
                              ref_labels=ref_labels)
            dsc_interim.append(dsc.cpu().numpy())
            index += 1
            if index == collection_point:
                dsc_all.append(np.mean(np.array(dsc_interim), axis=0))
                dsc_interim.clear()
                index = 0
            # may want to visualize the results (e.g. save it to the disk)

    dsc_all = np.array(dsc_all)
    mean_dsc_all = np.mean(dsc_all, axis=0)
    print(f"DSC for each test scan:")
    for dsc in mean_dsc_all:
        print(dsc)
    print()
    print(f"The average DSC is {np.mean(mean_dsc_all, axis=0)}")
