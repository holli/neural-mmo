import numpy as np
import time
import ray
import pickle
from collections import defaultdict

import projekt
from forge import trinity
from forge.trinity.timed import runtime

from forge.ethyr.io import Stimulus, Action, utils
from forge.ethyr.torch import optim
from forge.ethyr.experience import RolloutManager

# @ray.remote
# @ray.remote(num_gpus=1) # setting when calling so that we can switch easier
class God(trinity.God):
   '''Server level God API demo

   This server level optimizer node aggregates experience
   across all core level rollout worker nodes. It
   uses the aggregated experience compute gradients.

   This is effectively a lightweight variant of the
   Rapid computation model, with the potential notable
   difference that we also recompute the forward pass
   from small observation buffers rather than
   communicating large activation tensors.

   This demo builds up the ExperienceBuffer utility,
   which handles rollout batching.'''

   def __init__(self, trin, config, args, idx):
      '''Initializes a model and relevent utilities'''
      super().__init__(trin, config, args, idx)
      self.config, self.args = config, args

      self.manager = RolloutManager(config)

      self.net = projekt.ANN(config).to(self.config.DEVICE)

   @runtime
   def step(self, recv):
      '''Broadcasts updated weights to the core level
      Sword rollout workers. Runs rollout workers'''
      t = time.time()
      self.net.recvUpdate(recv)

      self.batch_size = []
      self.forward_item_timing = []
      self.backward_item_timing = []

      self._rollouts(recv)

      #Send update
      grads = self.net.grads()
      logs, nUpdates, nRollouts  = self.manager.reset()
      tb_logs = {'batch_size_array': np.array(self.batch_size),
                 'timing_forward_pass': np.mean(self.forward_item_timing),
                 'timing_backward_pass': np.mean(self.backward_item_timing),
                 'timing_total': time.time()-t }
      return grads, logs, nUpdates, nRollouts, tb_logs

   def _rollouts(self, recv):
      '''Runs rollout workers while asynchronously
      computing gradients over available experience'''
      done = False
      while not done:
         packets = super().distrib(recv) #async rollout workers
         self._processRollouts()          #intermediate gradient computatation
         packets = super().sync(packets) #sync next batches of experience
         self.manager.recv(packets)

         done = self.manager.nUpdates >= self.config.OPTIMUPDATES
      self._processRollouts() #Last batch of gradients

   def _processRollouts(self):
      '''Runs minibatch forwards/backwards
      over all available experience'''
      for batch in self.manager.batched(self.config.OPTIMBATCH, forOptim=True):
         t = time.time()
         rollouts = self.forward(*batch)
         self.forward_item_timing.append((time.time() -t)/len(rollouts))
         self.batch_size.append(len(rollouts))
         t = time.time()
         self.backward(rollouts)
         self.backward_item_timing.append((time.time() -t)/len(rollouts))

   def forward(self, pop, rollouts, data):
      '''Recompute forward pass and assemble rollout objects'''
      keys, _, stims, rawActions, actions, rewards, dones = data
      _, outs, vals = self.net(pop, stims, atnArgs=actions)

      #Unpack outputs
      atnTensor, idxTensor, atnKeyTensor, lenTensor = actions
      lens, lenTensor = lenTensor
      atnOuts = utils.unpack(outs, lenTensor, dim=1)

      #Collect rollouts
      for key, out, atn, val, reward, done in zip(
            keys, outs, rawActions, vals, rewards, dones):

         atnKey, lens, atn = list(zip(*[(k, len(e), idx)
            for k, e, idx in atn]))

         atn = np.array(atn)
         out = utils.unpack(out, lens)

         self.manager.fill(key, (atnKey, atn, out), val, done)

      return rollouts

   def backward(self, rollouts):
      '''Compute backward pass and logs from rollout objects'''
      reward, val, pg, valLoss, entropy = optim.backward(
            rollouts, device=self.config.DEVICE,
            valWeight=0.25, entWeight=self.config.ENTROPY)


