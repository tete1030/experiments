from collections import OrderedDict

__all__ = ["hparams", "config", "globalvars"]

class AttrDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self._update_subdict()

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

    def __setitem__(self, k, v):
        if not isinstance(v, AttrDict) and isinstance(v, dict):
            v = AttrDict(v)
        super(AttrDict, self).__setitem__(k, v)

    def _update_subdict(self):
        for key, value in self.items():
            if not isinstance(value, AttrDict) and isinstance(value, dict):
                self[key] = AttrDict(value)

    def update(self, srcdict):
        super(AttrDict, self).update(srcdict)
        self._update_subdict()

hparams = AttrDict()
config = AttrDict()
globalvars = AttrDict()
globalvars.main_context = AttrDict()
