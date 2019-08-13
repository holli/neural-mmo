import numpy as np
import torch
import time
from datetime import datetime

from collections import defaultdict

import projekt
from forge.ethyr.torch import save
from forge.ethyr.torch import Model
from forge.blade.lib.log import Quill

from forge import trinity
from forge.trinity.timed import runtime

class Pantheon(trinity.Pantheon):
   '''Cluster level Pantheon API demo

   This cluster level module aggregrates
   gradients across all server level optimizer
   nodes and updates model weights using Adam.

   Also demonstrates logging and snapshotting
   functionality through the Quill and Model
   libraries, respectively.'''

   def __init__(self, trinity, config, args):
      '''Initializes a copy of the model, which keeps
      track of a copy of the weights for the optimizer.'''
      super().__init__(trinity, config, args)
      self.config, self.args = config, args

      self.model = Model(config)
      self.quill = Quill(config.MODELDIR)
      self.log = defaultdict(list)
      self.tick = 0

      self.log_writer = self.config.log_writer()

   @runtime
   def step(self):
      '''Broadcasts updated weights to server level
      God optimizer nodes. Performs an Adam step
      once optimizers return a batch of gradients.'''

      # step_recvs = super().step(self.model.model)
      step_recvs = super().step(self.model.params.detach().numpy())

      #Write logs using Quill
      gradients_recvs, logs, nUpdates_list, nRollouts_list = list(zip(*step_recvs))

      nUpdates = sum(nUpdates_list)
      nRollouts = sum(nRollouts_list)
      self.quill.scrawl(logs, nUpdates, nRollouts)
      self.tick += 1

      if not self.config.TEST:
         self.model.step_optimizer(gradients_recvs, nUpdates, nRollouts)
         self.model.checkpoint(self.quill.lifetime)
         global_step = self.model.model_tick
      else:
         global_step = self.tick

      print(f'Model Tick: {self.model.model_tick}, Lifetime: {self.quill.lifetime:.2f},',
            f'Best: {self.model.lifetime_best:.2f}, Time: {datetime.utcnow()}')
      print(f'Updates (moves): Total: {self.model.updates_total}, Step: {nUpdates} |||',
            f'Rollouts (lives): Total: {self.model.rollouts_total}, Step: {nRollouts},')

      self.log_writer.add_scalar('nUpdates', nUpdates, global_step=global_step)
      self.log_writer.add_scalar('nRollouts', nRollouts, global_step=global_step)
      self.log_writer.add_histogram('nUpdatesHist', np.array(nUpdates_list), global_step=global_step)
      self.log_writer.add_histogram('nRolloutsHist', np.array(nRollouts_list), global_step=global_step)
      self.log_writer.add_scalar('Lifetime', self.quill.lifetime, global_step=global_step)
      self.log_writer.add_histogram('LifetimesHist', self.quill.lifetimes_arr, global_step=global_step)

      for key, vals in self.quill.lifetimes_ann.items():
         self.log_writer.add_scalar(f'LifetimePerAnn/{key}', np.mean(vals), global_step=global_step)


