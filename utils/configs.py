import torch

class States(object):
    
    def __init__(self, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_batch = -1
        self.curren_epoch = -1
        self.best_metric = dict(zip(cfg.EVAL.METRIC,cfg.EVAL.INITIAL_METRIC))

    def update(self, key, value=None, lam=None):
        '''Update or create the key with the value or the lambda.'''
        if lam:
            value = lam(self.__getattribute__(key))
        self.__setattr__(key,value)

    def step(self,key):
        if key in self.__dict__.keys():
            self.__setattr__(key, self.__getattribute__(key) + 1)
        else:
            self.update(key,0)
