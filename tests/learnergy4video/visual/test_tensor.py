import torch

from learnergy4video.visual import tensor


def test_show_tensor():
    t = torch.zeros(28, 28)

    tensor.show_tensor(t)
