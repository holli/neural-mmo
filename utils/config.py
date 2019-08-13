import os
from forge.blade.core import config
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np

from utils import global_consts # global_consts.config

class Config(config.Config):
   EXPERIMENTS_DIR = 'resource/exps'

   TEST = True
   RENDERING_WEB = False

   NENT = 128
   NPOP = 1

   NATN    = 1    #Number of actions taken by the network
   HIDDEN  = 128  #Model embedding dimension
   EMBED   = 128  #Model hidden dimension
   ENTROPY = 0.01 #Entropy bonus for policy gradient loss

   # OPTIMIZER_LR = 0.001
   OPTIMIZER_LR = 0.001
   OPTIMIZER_WEIGHT_DECAY = 0.00001

   # NGOD   = 2  #Number of GPU optimizer servers
   NGOD   = 1  #Number of GPU optimizer servers
   NSWORD = 2  #Number of CPU rollout workers per server

   #EPOCHUPDATES: Number of experience steps per
   #synchronized gradient step at the cluster level
   # EPOCHUPDATES = 2**14 #Training
   # EPOCHUPDATES = 2**8  #Local debug
   #EPOCHUPDATES = 2**2 # OHU TEST
   EPOCHUPDATES = 2**10 # OHU TRAIN

   #OPTIMUPDATES: Number of experience steps per
   #optimizer server per cluster level step
   #SYNCUPDATES: Number of experience steps between
   #syncing rollout workers to the optimizer server
   OPTIMUPDATES = EPOCHUPDATES / NGOD
   SYNCUPDATES  = OPTIMUPDATES / 2**4

   #OPTIMBATCH: Number of experience steps per
   #.backward minibatch on optimizer servers
   #SYNCUPDATES: Number of experience steps between
   #syncing rollout workers to the optimizer server
   OPTIMBATCH  = SYNCUPDATES * NGOD
   SYNCBATCH   = SYNCUPDATES

   RAY_MODE = 'default' # 'local' / 'default' / 'remote'

   #Device used on the God optimizer server. Rollout workers use CPU by default
   #Also used for ray to set num_gpus=1 if this is cuda
   DEVICE = 'cuda:0'
   # DEVICE = 'cpu'

   def log_writer(self):
      return SummaryWriter(self.TBLOGDIR_CURRENT)

   # Different gammas, you probably want to have it so that mean lifetime
   # results in learning from all steps
   # def test_gamma(gamma): return [f'{gamma**i:.2f}' for i in range(20)]
   # test_gamma(0.99) ['1.00', '0.99', '0.98', ... '0.83']
   # test_gamma(0.95) ['1.00', '0.95', '0.90', ... '0.38']
   # test_gamma(0.9)  ['1.00', '0.90', '0.81', ... '0.14']
   DISCOUNT_GAMMA = 0.99
   def reward_discount_array(self, N):
      discounts = np.array([self.DISCOUNT_GAMMA**i for i in range(N)])

      # discounts[20:] -= 0.2 # TODO USING MANUAL WEIRD REWARD; MAYBE SOMETHING BETTER??

      return discounts

   def _dump_yaml_to_versions(self):
      d_self = self.__dict__.copy()
      d_class = self.__class__.__dict__.copy()
      [d_class.pop(key, None) for key in d_self.keys()]
      d_class_base = self.__class__.__base__.__dict__.copy()
      [d_class_base.pop(key, None) for key in (list(d_self.keys()) + list(d_class.keys()))]

      data = {'self': d_self, 'class': d_class, 'class_base': d_class_base}

      vers_path = (Path(self.ROOT) / 'versions')
      vers_path.mkdir(parents=True, exist_ok=True)

      fname = vers_path / f'{self.VERSION:02d}.yml'
      with open(fname, 'w') as outfile:
         yaml.dump(data, outfile, sort_keys=False)

   def __init__(self, name, **kwargs):
      name = name.replace('.py', '')
      self.name = name

      self.ROOT = os.path.join(self.EXPERIMENTS_DIR, name, '')
      self.MODELDIR = os.path.join(self.ROOT, 'model')
      self.TBLOGDIR = os.path.join(self.ROOT, 'tb_log')

      Path(self.MODELDIR).mkdir(parents=True, exist_ok=True)

      version_file = Path(self.ROOT)/'version.txt'
      if not version_file.exists():
         self.VERSION = 0
      else:
         with version_file.open() as f: self.VERSION = int(f.readline()) + 1
      with open(version_file, 'w') as f: f.write(str(self.VERSION) + "\n")
      self.TBLOGDIR_CURRENT = os.path.join(self.TBLOGDIR, str(self.VERSION))

      if self.TEST:
         self.TBLOGDIR_CURRENT += '_test'

      Path(self.TBLOGDIR_CURRENT).mkdir(parents=True, exist_ok=True)

      # model_path = Path(self.MODELDIR)
      # for path in ['model', 'train', 'test', 'log']:
      # for path in ['model', 'log']:
      #    (model_path / path).mkdir(parents=True, exist_ok=True)
      #    print(model_path / path)

      for k, v in kwargs.items():
         setattr(self, k, v)

      self._dump_yaml_to_versions()
      global_consts.config = self

      print(f'Config: {self.name} ({self.VERSION}) --> NENT: {self.NENT}, NPOP: {self.NPOP} @ {self.ROOT}')
      print(f'Start tensorboard with\ntensorboard --logdir={self.TBLOGDIR}')

