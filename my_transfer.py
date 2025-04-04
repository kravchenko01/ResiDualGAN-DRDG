import sys
import PIL
import torchvision
from yacs.config import CfgNode as CN
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import logging

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import FloatTensor
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from core.models.residualgan import *
from core.models.resize_block import ResizeBlock
from core.models.build import *
from core.utils.utils import weights_init_normal, UnNormalize, adjust_param, setup_logger, BerHu
from core.utils.data_display import *
from core.datasets.dual_dataset import DualDataset
from core.configs.default import _C as cfg
from transfer import transfer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg",
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "-opts",
        help="Modify config options using the command-line",
        default="",
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("RDG", cfg.OUTPUT_DIR)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    # G_AB, _ = train(cfg)
    save_path = cfg.OUTPUT_DIR + "/models"
    model_path = f"{save_path}"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(f"{save_path}/temp", exist_ok=True)
    
    device = cfg.MODELS.DEVICE
    G_AB, G_BA = build_generators(cfg)
    G_AB.to(device)
    G_BA.to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)  
    
    G_AB.load_state_dict(torch.load(f"{model_path}/G_AB_0.pth"))
    G_BA.load_state_dict(torch.load(f"{model_path}/G_BA_0.pth"))
    D_A.load_state_dict(torch.load(f"{model_path}/D_A_0.pth"))
    D_B.load_state_dict(torch.load(f"{model_path}/D_B_0.pth"))

    transfer(G_AB, f"{cfg.DATASETS.SOURCE_PATH}", cfg.DATASETS.TARGET_SIZE,
             cfg.OUTPUT_DIR+"/data", torch.device(cfg.MODELS.DEVICE), batch_size=1)
    
    
if __name__ == "__main__":
    main()
    
