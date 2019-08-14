from pdb import set_trace as T
import numpy as np
import torch
import time
import os
import yaml
from datetime import datetime
from collections import defaultdict
from torch.nn.parameter import Parameter

from forge.ethyr.torch import save
from forge.ethyr.torch.optim import ManualAdam

import projekt

class Model:
   '''Model manager class

   Convenience class wrapping saving/loading,
   model initialization, optimization, and logging.

   Args:
      ann: Model to optimize. Used to initialize weights.
      config: A Config specification
      args: Hook for additional user arguments
   '''
   def __init__(self, config):
      # self.saver = save.Saver(config.MODELDIR, 'models', 'bests', resetTol=256)
      self.config = config

      # self.models = ann(self.config).params()
      ann = projekt.ANN(self.config)
      ann_params = ann.params()
      self.params = Parameter(torch.Tensor(np.array(ann_params)))

      self.model_tick = 0
      self.time_start = time.time()
      self.time_checkpoint = self.time_start
      self.lifetime_best = 0
      self.rollouts_total = 0
      self.updates_total = 0

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
   def step_optimizer(self, gradients, updates_increase, rollouts_increase):
      '''Clip the provided gradients and step the optimizer

      Args:
         gradList: a list of gradients
      '''
      self.model_tick += 1
      self.rollouts_total += rollouts_increase
      self.updates_total += updates_increase

      grad = np.array(gradients)
      grad = np.mean(grad, 0)
      grad = np.clip(grad, -5, 5)

      gradAry = torch.Tensor(grad)
      self.optimizer.step(gradAry)

   def checkpoint(self, lifetime, extra_info = {}):
      assert not self.config.TEST
      lifetime = float(lifetime)

      info = {}
      best = lifetime > self.lifetime_best
      if best:
         self.lifetime_best = lifetime
      info['lifetime'] = lifetime
      info['lifetime_best'] = self.lifetime_best
      info['model_tick'] = self.model_tick
      info['config_version'] = self.config.VERSION
      info['time_utc']: str(datetime.utcnow())
      info['time_start'] = int(self.time_start)
      info['time_checkpoint'] = int(time.time())
      info['time_duraction_checkpoint'] = int(time.time() - self.time_checkpoint)
      info['rollouts_total'] = self.rollouts_total
      info['updates_total'] = self.updates_total

      for key, value in extra_info.items():
         info[key] = value

      data = {'param': self.params, 'optimizer': self.optimizer.state_dict(), 'info': info}

      if best:
         self._save('bests', data)

      #if self.epoch % 100 == 0:
      saving_tick_checkpoint = self.model_tick % 10 == 0
      # saving_tick_checkpoint = self.model_tick % 50 == 0
      if saving_tick_checkpoint or info['time_duraction_checkpoint'] > 60*60: # minimum once a hour
         self._save(f'model_{self.model_tick:04d}', data)

      self.time_checkpoint = time.time()

   def _save(self, fname, data):
      path = os.path.join(self.config.MODELDIR, fname)
      torch.save(data, path + '.pth')

      # Only for easier browsing of files
      with open(path + ".yml", 'w') as outfile:
         yaml.dump(data['info'], outfile, sort_keys=False)
         # json.dump(data['info'], outfile)
         # outfile.write("\n")

   def load_best(self):
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
         self.lifetime_best = float(info['lifetime'])
         if 'rollouts_total' in info: # supporting older saves
            self.rollouts_total = info['rollouts_total']
            self.updates_total = info['updates_total']

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

