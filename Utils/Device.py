import torch


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"