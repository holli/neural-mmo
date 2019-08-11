# from pdb import set_trace as T
# import numpy as np
# import torch
# import time
# from forge.blade.lib.utils import EDA

# import os
# from datetime import datetime
# import json

# class Resetter:
#    '''Utility for model stalling that keeps track
#    of the time since the model has improved

#    Args:
#       resetTol: ticks before resetting
#    '''
#    def __init__(self, resetTol):
#       self.resetTicks, self.resetTol = 0, resetTol

#    def step(self, best=False):
#       '''Step the resetter

#       Args:
#          best (bool): whether this is the best model
#             performance thus far

#       Returns:
#          bool: whether the model should be reset to
#             the previous best checkpoint
#       '''
#       if best:
#          self.resetTicks = 0
#       elif self.resetTicks < self.resetTol:
#          self.resetTicks += 1
#       else:
#          self.resetTicks = 0
#          return True
#       return False

# class Saver:
#    '''Model save/load class

#    Args:
#       root: Path for save
#       savef: Name of the save file
#       bestf: Name of the best checkpoint file
#       resetTol: How often to reset if training stalls
#    '''
#    def __init__(self, root, savef, bestf, resetTol):
#       self.bestf, self.savef = bestf, savef,
#       self.root, self.extn = root, '.pth'

#       self.resetter = Resetter(resetTol)
#       self.rewardAvg, self.best = EDA(), 0
#       self.start, self.epoch = time.time(), 0
#       self.resetTol = resetTol

#    def save(self, params, opt, fname):
#       data = {'param': params,
#               'opt' : opt.state_dict(),
#               'epoch': self.epoch}
#       # torch.save(data, self.root + fname + self.extn)
#       path = os.path.join(self.root, fname)
#       torch.save(data, path + self.extn)

#       info = {'epoch': self.epoch, 'reward': self.reward, 'best_so_far': self.best, 'time': self.time, 'utc': str(datetime.utcnow()) }
#       with open(path + ".txt", 'w') as outfile:
#          json.dump(info, outfile)
#          outfile.write("\n")

#    # def checkpoint(self, params, opt, reward):
#    #    '''Save the model to file

#    #    Args:
#    #       params: Parameters to save
#    #       opt: Optimizer to save
#    #       fname: File to save to
#    #    '''

#    #    self.time  = time.time() - self.start
#    #    self.start = time.time()
#    #    self.reward = reward
#    #    self.epoch += 1

#    #    self.save(params, opt, self.savef)

#    #    best = reward > self.best
#    #    if best:
#    #       self.best = reward
#    #       self.save(params, opt, self.bestf)

#    #    #if self.epoch % 100 == 0:
#    #    if self.epoch % 50 == 0:
#    #       self.save(params, opt, 'model_'+str(self.epoch))

#    #    return self.resetter.step(best)

#    # def load(self, params, opt, best=False):
#    #    '''Load the model from file

#    #    Args:
#    #       opt: Optimizer to load
#    #       params: Parameters to load
#    #       best: Whether to load the best or latest checkpoint

#    #    Returns:
#    #       info: {model_tick, epoch, lifetime, config_version, time}
#    #    '''
#    #    fname = self.bestf if best else self.savef
#    #    path  = os.path.join(self.root, fname) + self.extn
#    #    data  = torch.load(path)
#    #    params.data = data['param']
#    #    if opt is not None:
#    #       opt.load_state_dict(data['opt'])

#    #    return data['extra']

#    def print(self):
#       '''Print stats for the latest epoch'''
#       print(
#             'Tick: ', self.epoch,
#             ', Time: ', str(self.time)[:5],
#             ', Lifetime: ', str(self.reward)[:5],
#             ', Best: ', str(self.best)[:5])

