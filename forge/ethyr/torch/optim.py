import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from pdb import set_trace as T

from collections import defaultdict
from forge.ethyr.torch import loss

class ManualAdam(optim.Adam):
   '''Adam wrapper that accepts gradient lists'''
   def step(self, grads):
      '''Takes an Adam step on the parameters using
      the user provided gradient list provided.

      Args:
         grads: A list of gradients
      '''
      grads = Variable(torch.Tensor(np.array(grads)))
      self.param_groups[0]['params'][0].grad = grads
      super().step()

class ManualSGD(optim.SGD):
   '''SGD wrapper that accepts gradient lists'''
   def step(self, grads):
      '''Takes an SGD step on the parameters using
      the user provided gradient list provided.

      Args:
         grads: A list of gradients
      '''
      grads = Variable(torch.Tensor(np.array(grads)))
      self.param_groups[0]['params'][0].grad = grads
      super().step()

def merge(rollouts):
   '''Merges all collected rollouts for batched
   compatibility with optim.backward'''

   outs = {'value': [], 'return': [],
         'action': defaultdict(lambda: defaultdict(list))}
   for rollout in rollouts.values():
      for idx in range(rollout.time):
         try:
            key, atn, out = rollout.outs[idx]
         except:
            print(rollout.time)
            print(len(rollout))
            print(len(rollout.returns))
            print(len(rollout.outs))
            print('----')
            T()

         val = rollout.vals[idx]
         ret = rollout.returns[idx]

         outs['value'].append(val)
         outs['return'].append(ret)

         for k, o, a in zip(key, out, atn):
            k = tuple(k)
            outk = outs['action'][k]
            outk['atns'].append(o)
            outk['idxs'].append(a)
            outk['vals'].append(val)
            outk['rets'].append(ret)
   return outs


def backward(rollouts, valWeight=0.5, entWeight=0, device='cpu'):
   '''Computes gradients from a list of rollouts

   Args:
      rolls: A list of rollouts
      valWeight (float): Scale to apply to the value loss
      entWeight (float): Scale to apply to the entropy bonus
      device (str): Hardware to run backward on

   Returns:
      reward: Mean reward achieved across rollouts
      val: Mean value function estimate across rollouts
      pg: Policy gradient loss
      valLoss: Value loss
      entropy: Entropy bonus
   '''
   outs = merge(rollouts)
   pg, entropy, attackentropy = 0, 0, 0
   for k, out in outs['action'].items():
      atns = out['atns']
      vals = torch.stack(out['vals']).to(device)
      idxs = torch.tensor(out['idxs']).to(device)
      rets = torch.tensor(out['rets']).to(device).view(-1, 1)
      l, e = loss.PG(atns, idxs, vals, rets)
      pg += l
      entropy += e

   returns = torch.stack(outs['value']).to(device)
   values  = torch.tensor(outs['return']).to(device).view(-1, 1)
   valLoss = loss.valueLoss(values, returns)
   totLoss = pg + valWeight*valLoss + entWeight*entropy

   totLoss.backward()
   reward = np.mean(outs['return'])

   return reward, vals.mean(), pg, valLoss, entropy

