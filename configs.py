from easydict import EasyDict as edict
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
cfg = edict()
cfg.device_ids = [0, 1]
cfg.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


cfg.datasets = edict()
# cfg.datasets.data_path = "../data/training"
cfg.datasets.data_path = "../registration_bone_data"

cfg.train = edict()
cfg.train.multiGPU = True

cfg.train.device_ids = cfg.device_ids
cfg.train.device = cfg.device

cfg.train.optimizer = "Adam"
cfg.train.scheduler = "CosineAnnealingLR"
cfg.train.lr = 1e-5
cfg.train.gamma = 0.9

cfg.train.batch_size = 2
cfg.train.num_workers = 4
cfg.train.max_epochs = 50
cfg.train.valid_interval = 1


cfg.train.save_path = "./train_result_bone"
