from utils import global_consts
from pdb import set_trace as T
import ray
import pickle
import time

from forge.trinity.timed import Timed, runtime, waittime

#Environment logic
class God(Timed):
   '''A simple Server level interface for generic,
   persistent, and asynchronous computation over
   remote Cores (Sword API)

   Args:
      trinity: A Trinity object
      config: A forge.blade.core.Config object
      args: Hook for additional user arguments
      idx: An index specifying the current server
   '''
   def __init__(self, trinity, config, args, idx):
      super().__init__()
      if global_consts.RAY_REMOTE_SWORD:
         self.disciples = [trinity.sword.remote(trinity, config, args, i+idx*config.NGOD) for i in range(config.NSWORD)]
      else:
         self.disciples = [trinity.sword._modified_class(trinity, config, args, i+idx*config.NGOD) for i in range(config.NSWORD)]

   def distrib(self, packet=None):
      '''Asynchronous wrapper around the step
      function of all remote Cores (Sword API)

      Args:
         packet: Arbitrary user data broadcast
            to all Cores (Sword API)

      Returns:
         A list of async handles to the step returns
         from all remote cores (Sword API)
      '''
      rets = []
      for sword in self.disciples:
         if global_consts.RAY_REMOTE_SWORD:
            rets.append(sword.step.remote(packet))
         else:
            rets.append(sword.step(packet))
      return rets

   def step(self, packet=None):
      '''Synchronous wrapper around the step
      function of all remote Cores (Sword API)

      Args:
         packet: Arbitrary user data broadcast
            to all Cores (Sword API)

      Returns:
         A list of step returns from
         all remote cores (Sword API)
      '''
      rets = self.distrib(packet)
      return self.sync(rets)

   @waittime
   def sync(self, rets):
      '''Synchronizes returns from distrib

      Args:
         rets: async handles returned from distrib

      Returns:
         A list of step returns from
         all remote cores (Sword API)
      '''
      if global_consts.RAY_REMOTE_SWORD:
         return ray.get(rets)
      else:
         return rets
