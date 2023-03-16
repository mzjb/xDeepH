#!/usr/bin/env python
# ========================= #
# Training of network model #
# ========================= #

# Usage: python <path-to-this-file>/train.py <your_config>.ini [-n NUM_THREADS]
# Default train config file is xdeeph/default_configs/train_default.ini

import os
import argparse

parser = argparse.ArgumentParser(description='Train xDeepH network')
parser.add_argument('config', type=str, metavar='CONFIG', help='Config file for training')
parser.add_argument('-n', type=int, default=None, help='Maximum number of threads')
args = parser.parse_args()

if args.n is not None:
    os.environ["OMP_NUM_THREADS"] = f"{args.n}"
    os.environ["MKL_NUM_THREADS"] = f"{args.n}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{args.n}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{args.n}"
    os.environ["VECLIB_MAXIMUM_THREADS"] = f"{args.n}"
    import torch
    torch.set_num_threads(args.n)

from xdeeph import xDeepHKernel
kernel = xDeepHKernel()
kernel.train(args.config)
