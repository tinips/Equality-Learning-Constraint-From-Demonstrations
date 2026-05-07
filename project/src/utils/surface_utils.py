import numpy as np
import torch

from src.models.models3D import SpiralImplicitMLP


def load_learned_surface(model_path):
    model = SpiralImplicitMLP()
    state = torch.load(model_path)
    model.load_state_dict(state)
    model.eval()

    return model
