import argparse
import itertools
import os
import random
import warnings

import numpy as np
from torch.optim import lr_scheduler

from model.CIBUR_MAN import CIBUR
from utils.manh_data import *
from utils.configure import get_default_config

parser = argparse.ArgumentParser()


parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='1', help='gap of print evaluations')

args = parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    config = get_default_config()
    config['print_num'] = args.print_num

    seed = config['training']['seed']
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True

    model = CIBUR()
    optimizer = torch.optim.Adam(
        itertools.chain(model.encoder_z1.parameters(),
                        model.encoder_z2.parameters(),
                        model.encoder_z3.parameters(),
                        model.autoencoder_a.parameters(),
                        model.autoencoder_b.parameters(),
                        model.autoencoder_c.parameters(),
                        model.mi_estimator_soft1.parameters(),
                        model.pr1.parameters(),
                        model.pr2.parameters()
                        ), lr=config['training']['lr'], weight_decay=1e-6)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3, verbose=False)

    model.to_device(device)

    redata = ReData()
    xs_raw = [redata.a_m, redata.s_m, redata.d_m]
    xs = []
    for view in range(len(xs_raw)):
        xs.append(torch.from_numpy(xs_raw[view]).float().to(device))

    model.train(config, xs, optimizer, scheduler)


if __name__ == '__main__':
    main()
