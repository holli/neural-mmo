from pdb import set_trace as T
import ray
import pickle
import time

from forge.trinity.timed import Timed, runtime, waittime
from utils import global_consts

#Cluster/Master logic
class Pantheon(Timed):
   '''A simple Cluster level interface for generic,
   persistent, and asynchronous computation over
   multiple remote Servers (God API)

   Args:
      trinity: A Trinity object
      config: A forge.blade.core.Config object
      args: Hook for additional user arguments
      remote_god: If you want to run god locally for debugging
   '''
   def __init__(self, trinity, config, args):
      super().__init__()
      if global_consts.RAY_REMOTE_GOD:
         self.disciples = [trinity.god.remote(trinity, config, args, idx) for idx in range(config.NGOD)]
      else:
         self.disciples = [trinity.god._modified_class(trinity, config, args, 0)]

   def distrib(self, packet):
      '''Asynchronous wrapper around the step function
      function of all remote Servers (God level API)

      Args:
         packet: Arbitrary user data broadcast
            to all Servers (God API)

      Returns:
         A list of async handles to the step returns
         from all remote servers (God API)
      '''
      rets = []
      for god in self.disciples:
         rets.append(god.step.remote(packet))
      return rets

   def step(self, packet=None):
      '''Synchronous wrapper around the step function
      function of all remote Servers (God level API)

      Args:
         packet: Arbitrary user data broadcast
            to all Servers (God API)

      Returns:
         A list of step returns from all
         remote servers (God level API)
      '''
      if global_consts.RAY_REMOTE_GOD:
         rets = self.distrib(packet)
         return self.sync(rets)
      else:
         return [self.disciples[0].step(packet)]

   @waittime
   def sync(self, rets):
      '''Synchronizes returns from distrib

      Args:
         rets: async handles returned from distrib

      Returns:
         A list of step returns from all
         remote servers (God API)
      '''
      return ray.get(rets)


