import argparse
from utils import global_consts
from forge.ethyr.torch import Model

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

   config.TEST = True
   config.RENDERING_WEB = True

   #    global_consts.RAY_REMOTE_GOD = False

   if global_consts.RAY_REMOTE_GOD:
      sword = trin.sword.remote(trin, config, args, idx=0)
      env = sword.getEnv.remote()

      model = Model(config)
      sword.step.remote(model.params.detach().numpy())

      Application(env, sword.tick.remote)
   else:
      print("Starting with local stuff")
      sword = trin.sword._modified_class(trin, config, args, idx=0)
      env = sword.getEnv()
      model = Model(config)
      sword.net.recvUpdate(model.params.detach().numpy())
      Application(env, sword.tick)


