import torch
from src.explore import lidar_check, cumsum_check
from src.models import compile_model
from src.data import NuscData

import fire

if __name__ == "__main__":
    lidar_check()