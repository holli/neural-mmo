
import argparse
# from pdb import set_trace as T

# from experiments import Experiment# , Config
# from forge.blade import lib

# from forge.trinity import smith, Trinity
# from forge.trinity.timed import TimeLog

# from projekt import Pantheon, God, Sword

# from pdb import set_trace as T
# import os

# from utils.run_utils import *

# from forge.blade.core import config


def parse_args():
  '''Processes command line arguments'''
  parser = argparse.ArgumentParser('Projekt Godsword')
  parser.add_argument('--ray', type=str, default='default',
        help='Ray mode (local/default/remote)')
  parser.add_argument('--render', action='store_true', default=False,
        help='Render env')
  return parser.parse_args()

def render(trin, config, args):
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

  from forge.embyr.twistedserver import Application
  sword = trin.sword.remote(trin, config, args, idx=0)
  env = sword.getEnv.remote()
  Application(env, sword.tick.remote)


