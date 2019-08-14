import argparse
from forge.ethyr.torch import Model
from forge.blade import lib
from forge.trinity import smith, Trinity
from forge.trinity.timed import TimeLog
from projekt import Pantheon, God, Sword
import ray
import torch

def GetGodRay(config):
   # return God
   if 'cuda' in config.DEVICE.lower():
      return ray.remote(num_gpus=1)(God)
   else:
      return ray.remote(num_cpus=1)(God)

def train_loop(config, args):
   lib.ray.init(config.RAY_MODE)
   # lib.ray.init('local') # Might not work in all cases, caused some problems with grad transfers

   #Create a Trinity object specifying: Cluster, Server, and Core level execution
   trinity = Trinity(Pantheon, GetGodRay(config), Sword)
   # trinity = Trinity(Pantheon, God, Sword)
   trinity.init(config, args)

   while True:
      trinity.step()
      TimeLog.log(trinity.logs())

def parse_args():
   '''Processes command line arguments'''
   parser = argparse.ArgumentParser('Projekt Godsword')
   parser.add_argument('--ray', type=str, default='default',
      help='Ray mode (local/default/remote)')
   parser.add_argument('--render', action='store_true', default=False,
      help='Render env')
   return parser.parse_args()

def render(config, args):
   """Runs the environment in render mode

   Connect to localhost:8080 to view the client.

   Args:
      trin   : A Trinity object as shown in __main__
      config : A Config object as shown in __main__

   Notes:
      Blocks execution. This is an unavoidable side
      effect of running a persistent server with
      a fixed tick rate
      """

   lib.ray.init('local')
   trinity = Trinity(Pantheon, GetGodRay(config), Sword)

   from forge.embyr.twistedserver import Application

   config.TEST = True
   config.RENDERING_WEB = True


   sword = trin.sword.remote(trin, config, args, idx=0)
   env = sword.getEnv.remote()
   model = Model(config)
   sword.step.remote(model.params.detach().numpy())
   Application(env, sword.tick.remote)



