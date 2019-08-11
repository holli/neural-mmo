from pdb import set_trace as T
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

      self.model = Model(projekt.ANN, config)
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
      recvs, logs, nUpdates, nRollouts = list(zip(*step_recvs))

      self.quill.scrawl(logs, sum(nUpdates), sum(nRollouts))
      self.tick += 1
      self.quill.print()

      if not self.config.TEST:
         self.model.stepOpt(recvs)
         self.model.checkpoint(self.quill.lifetime)

         print(f'Model Tick: {self.model.model_tick}, Lifetime: {self.quill.lifetime}, Best: {self.model.lifetime_best}, Time: {datetime.utcnow()}')

         global_step = self.model.model_tick
         self.log_writer.add_scalar('nUpdates', sum(nUpdates), global_step=global_step)
         self.log_writer.add_scalar('nRollouts', sum(nRollouts), global_step=global_step)
         self.log_writer.add_histogram('nUpdatesHist', np.array(nUpdates), global_step=global_step)
         self.log_writer.add_histogram('nRolloutsHist', np.array(nRollouts), global_step=global_step)
         self.log_writer.add_scalar('Lifetime', self.quill.lifetime, global_step=global_step)
         self.log_writer.add_histogram('LifetimesHist', self.quill.lifetimes_arr, global_step=global_step)

         for key, vals in self.quill.lifetimes_ann.items():
            self.log_writer.add_scalar(f'LifetimePerAnn/{key}', np.mean(vals), global_step=global_step)


