from collections import OrderedDict

__all__ = ["hparams", "config", "globalvars"]

class AttrDict(dict):
    def __getattr__(self, k):
        return self.__getitem__(k)

    def __setattr__(self, k, v):
        self.__setitem__(k, v)

    def __delattr__(self, k):
        self.__delitem__(k)

    def get(self, k, default=None):
        if k in self:
            return self.__getitem__(k)
        else:
            return default

    def set(self, k, v):
        self.__setitem__(k, v)

hparams = OrderedDict()
config = AttrDict()
globalvars = AttrDict()
