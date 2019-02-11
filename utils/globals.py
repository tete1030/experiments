import copy
from collections import OrderedDict
from types import SimpleNamespace

__all__ = ["hparams", "config", "globalvars"]

class AttrDict(OrderedDict):
    def __init__(self, *args, recursive_convert=True, copy=True, **kwargs):
        self._recursive_convert = recursive_convert
        self._copy = copy
        super(AttrDict, self).__init__(*args, **kwargs)
        self._update_subdict()

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError(k)
        else:
            return self.__getitem__(k)

    def __setattr__(self, k, v):
        if k.startswith("_"):
            object.__setattr__(self, k, v)
        else:
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
        if self._copy:
            v = copy.deepcopy(v)
        if self._recursive_convert and not isinstance(v, AttrDict) and isinstance(v, dict):
            v = AttrDict(v, copy=False)
        super(AttrDict, self).__setitem__(k, v)

    def _update_subdict(self, root=None):
        if root is None:
            _root = self
        elif self._copy:
            _root = copy.deepcopy(root)
        else:
            _root = root

        if self._recursive_convert:
            for key, value in _root.items():
                if not isinstance(value, AttrDict) and isinstance(value, dict):
                    _root[key] = AttrDict(value, copy=False)

        if root is not None:
            return _root

    def update(self, srcdict):
        srcdict = self._update_subdict(srcdict)
        super(AttrDict, self).update(srcdict)

class Namespace(SimpleNamespace):
    def get(self, k, default=None):
        return getattr(self, k, default)

hparams = AttrDict()
config = AttrDict()
globalvars = AttrDict()
globalvars.main_context = Namespace()
