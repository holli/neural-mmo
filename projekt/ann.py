'''Demo agent class'''
from pdb import set_trace as T
import numpy as np
import torch

from torch import nn

from forge import trinity

from forge.blade.lib.enums import Neon
from forge.blade.lib import enums
from forge.ethyr import torch as torchlib

from forge.ethyr.torch import policy
from forge.blade import entity

from forge.ethyr.torch.param import setParameters, getParameters, zeroGrads
from forge.ethyr.torch import param

from forge.ethyr.torch.io.stimulus import Env
from forge.ethyr.torch.io.action import NetTree
from forge.ethyr.torch.policy import attention

class Net(nn.Module):
   def __init__(self, config):
      super().__init__()

      h = config.HIDDEN
      net = attention.MiniAttend
      #net = attention.MaxReluBlock
      self.attn1 = net(h)
      self.attn2 = net(h)
      self.val  = torch.nn.Linear(h, 1)

class ANN(nn.Module):
   def __init__(self, config):
      super().__init__()
      self.config = config
      self.net = nn.ModuleList([Net(config) for _ in range(config.NPOP)])

      #Shared environment/action maps
      self.env    = Env(config)
      self.action = NetTree(config)

   #TODO: Need to select net index
   def forward(self, pop, stim_input, obs=None, atnArgs=None):
      # stim_input['Entity'][0] => [[(0, 1, 0, 1, 0)]]
      # stim_input['Entity'][1] => defaultdict(<class 'list'>, {'Food': array([[32]]), 'Water': array([[32]]), 'Health': array([[10]]), 'TimeAlive': array([[0]]), 'Damage': array([[0]]), 'Freeze': array([[0]]), 'Immune': array([[15]]), 'Self': array([[1]]), 'Population': array([[0]]), 'R': array([[0]]), 'C': array([[0]]), 0: [], 1: []})
      # len(stim_input['Tile'][0][0]) => 225
      # stim_input['Tile'][0] => [[(0, 1, 2, 41, 1), (0, 1, 2, 42, 1), .....
      # stim_input['Tile'][1].keys() => dict_keys(['NEnts', 'Index', 'RRel', 'CRel'])
      # NEnts => number of entities, Index => Type of tile, RRel ja CRel => column ja row index,
      # len(stim_input['Tile'][1]['NEnts'][0]) => 225

      net = self.net[pop]
      # sum([param.numel() for param in net.parameters() if param.requires_grad]) => 66177

      stim, embed = self.env(net, stim_input)
      # embed[0].values() => dict_values([0, 1, 2, ... 245
      # embed[0] values on erikoiset
         # embed[0][forge.blade.io.action.static.Melee] => 197
         # embed[0][(0, 1, 2, 41, 1)] => 0
      # embed[1].shape => torch.Size([246, 128])
      val         = net.val(stim)

      atnArgs, outs = self.action(stim, embed, obs, atnArgs)
      # import ipdb; ipdb.set_trace()
      return atnArgs, outs, val

   def recvUpdate(self, update):
      if update is None:
         return

      setParameters(self, update)
      zeroGrads(self)

   def grads(self):
      return param.getGrads(self)

   def params(self):
      return param.getParameters(self)

   #Messy hooks for visualizers
   def visDeps(self):
      from forge.blade.core import realm
      from forge.blade.core.tile import Tile
      colorInd = int(self.config.NPOP*np.random.rand())
      color    = Neon.color12()[colorInd]
      color    = (colorInd, color)
      ent = realm.Desciple(-1, self.config, color).server
      targ = realm.Desciple(-1, self.config, color).server

      sz = 15
      tiles = np.zeros((sz, sz), dtype=object)
      for r in range(sz):
         for c in range(sz):
            tiles[r, c] = Tile(enums.Grass, r, c, 1, None)

      targ.pos = (7, 7)
      tiles[7, 7].addEnt(0, targ)
      posList, vals = [], []
      for r in range(sz):
         for c in range(sz):
            ent.pos  = (r, c)
            tiles[r, c].addEnt(1, ent)
            #_, _, val = self.net(tiles, ent)
            val = np.random.rand()
            vals.append(float(val))
            tiles[r, c].delEnt(1)
            posList.append((r, c))
      vals = list(zip(posList, vals))
      return vals

   def visVals(self, food='max', water='max'):
      from forge.blade.core import realm
      posList, vals = [], []
      R, C = self.world.shape
      for r in range(self.config.BORDER, R-self.config.BORDER):
          for c in range(self.config.BORDER, C-self.config.BORDER):
            colorInd = int(self.config.NPOP*np.random.rand())
            color    = Neon.color12()[colorInd]
            color    = (colorInd, color)
            ent = entity.Player(-1, color, self.config)
            ent._r.update(r)
            ent._c.update(c)
            if food != 'max':
               ent._food = food
            if water != 'max':
               ent._water = water
            posList.append(ent.pos)

            self.world.env.tiles[r, c].addEnt(ent.entID, ent)
            stim = self.world.env.stim(ent.pos, self.config.STIM)
            #_, _, val = self.net(stim, ent)
            val = np.random.rand()
            self.world.env.tiles[r, c].delEnt(ent.entID)
            vals.append(float(val))

      vals = list(zip(posList, vals))
      return vals
