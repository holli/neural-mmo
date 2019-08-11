import os
from forge.blade.core import config
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class Config(config.Config):
   EXPERIMENTS_DIR = 'resource/exps'
   # MODELDIR = 'resource/exps' #Where to store models

   # LOAD = True #Load model from file?
   # BEST = True #If loading, most recent or highest lifetime?
   # TEST = True #Update the model during run?

   # LOAD = False
   # BEST = False
   # TEST = False

   LOAD = True
   BEST = True
   TEST = True

   NENT = 128
   NPOP = 1

   NATN    = 1    #Number of actions taken by the network
   HIDDEN  = 128  #Model embedding dimension
   EMBED   = 128  #Model hidden dimension
   ENTROPY = 0.01 #Entropy bonus for policy gradient loss

   # OPTIMIZER_LR = 0.001
   OPTIMIZER_LR = 0.01
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

   #Device used on the optimizer server.
   #Rollout workers use CPU by default
   DEVICE = 'cuda:0'

   def log_writer(self):
      return SummaryWriter(self.TBLOGDIR_CURRENT)

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
      Path(self.TBLOGDIR_CURRENT).mkdir(parents=True, exist_ok=True)

      # model_path = Path(self.MODELDIR)
      # for path in ['model', 'train', 'test', 'log']:
      # for path in ['model', 'log']:
      #    (model_path / path).mkdir(parents=True, exist_ok=True)
      #    print(model_path / path)

      for k, v in kwargs.items():
         setattr(self, k, v)

      print(f'Config: {self.name} ({self.VERSION}) --> NENT: {self.NENT}, NPOP: {self.NPOP} @ {self.ROOT}')
      print(f'Start tensorboard with\ntensorboard --logdir={self.TBLOGDIR}')

