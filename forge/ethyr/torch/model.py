from pdb import set_trace as T
import numpy as np
import torch
import time
import os
import json
from collections import defaultdict
from torch.nn.parameter import Parameter

from forge.ethyr.torch import save
from forge.ethyr.torch.optim import ManualAdam

class Model:
   '''Model manager class

   Convenience class wrapping saving/loading,
   model initialization, optimization, and logging.

   Args:
      ann: Model to optimize. Used to initialize weights.
      config: A Config specification
      args: Hook for additional user arguments
   '''
   def __init__(self, ann, config):
      # self.saver = save.Saver(config.MODELDIR, 'models', 'bests', resetTol=256)
      self.config = config

      # self.models = ann(self.config).params()
      ann_params = ann(self.config).params()
      self.params = Parameter(torch.Tensor(np.array(ann_params)))

      self.model_tick = 0
      self.time_start = time.time()
      self.time_checkpoint = self.time_start
      self.lifetime_best = 0

      if self.config.TEST:
         self.optimizer = None
      else:
         self.optimizer = ManualAdam([self.params], lr=self.config.OPTIMIZER_LR, weight_decay=self.config.OPTIMIZER_WEIGHT_DECAY)

      load_info = self.load_best()
      if load_info:
         print('Loaded model weights:', load_info)
      else:
         print("Could not load net/model from file. Using randomized weights.")

      # self.init(ann)
      # if self.config.LOAD or self.config.BEST:
      #    self.load(self.config.BEST)

   # #Initialize a new network
   # def initModel(self, ann):
   #    # self.models = ann(self.config).params()
   #    ann_params = ann(self.config).params()
   #    self.params = Parameter(torch.Tensor(np.array(ann_params)))

   #Grads and clip
   def stepOpt(self, gradList):
      '''Clip the provided gradients and step the optimizer

      Args:
         gradList: a list of gradients
      '''
      self.model_tick += 1
      grad = np.array(gradList)
      grad = np.mean(grad, 0)
      grad = np.clip(grad, -5, 5)

      gradAry = torch.Tensor(grad)
      self.optimizer.step(gradAry)

   def checkpoint(self, lifetime, extra_info = {}):
      '''Save the model to checkpoint

      Args:
         reward: Mean reward of the model
      '''
      assert not self.config.TEST

      info = {}
      info['time_utc']: str(datetime.utcnow())
      info['time_start'] = self.time_start
      info['time_checkpoint'] = time.time()
      info['time_duraction_checkpoint'] = self.time_checkpoint - time.time()
      info['config_version'] = self.config.VERSION
      info['model_tick'] = self.model_tick
      best = lifetime > self.lifetime_best
      if best:
         self.lifetime_best = lifetime
      info['lifetime'] = lifetime
      info['lifetime_best'] = self.lifetime_best

      for key, value in extra_info.items():
         info[key] = value

      data = {'param': self.params, 'optimizer': self.optimizer.state_dict(), 'info': info}

      if best:
         self._save('bests', data)

      #if self.epoch % 100 == 0:
      saving_tick_checkpoint = self.model_tick % 10 == 0
      # saving_tick_checkpoint = self.model_tick % 50 == 0
      if saving_tick_checkpoint or info['time_duraction_checkpoint'] > 60*60: # minimum once a hour
         self._save('model_'+str(self.model_tick), data)

      self.time_checkpoint = time.time()

   def _save(self, fname, data):
      # torch.save(data, self.root + fname + self.extn)
      path = os.path.join(self.config.MODELDIR, (fname + '.pth'))
      torch.save(data, path)

      with open(path + ".txt", 'w') as outfile:
         json.dump(data['info'], outfile)
         outfile.write("\n")

   def load_best(self):
      '''Load a model from file

      Args:
         best (bool): Whether to load the best (True)
             or most recent (False) checkpoint
      '''

      fname = os.path.join(self.config.MODELDIR, 'bests.pth')
      # fname = self.bestf if best else self.savef
      if not os.path.isfile(fname):
         return False
      data = torch.load(fname)

      self.params.data = data['param']
      if self.optimizer is not None:
         self.optimizer.load_state_dict(data['optimizer'])

      info = data['info']
      if info:
         self.model_tick = info['model_tick']
         self.lifetime_best = info['lifetime']

      return info

   # @property
   # def nParams(self):
   #    '''Print the number of model parameters'''
   #    nParams = len(self.model)
   #    print('#Params: ', str(nParams/1000), 'K')

   # @property
   # def model(self):
   #    '''Get model parameters

   #    Returns:
   #       a numpy array of model parameters
   #    '''
   #    return self.params.detach().numpy()

