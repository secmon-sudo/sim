import sys
import importlib

sys.modules['dummy'] = type('module', (), {'__name__': 'dummy'})
import dummy

for k in list(sys.modules.keys()):
    if 'dummy' in k:
        del sys.modules[k]

import dummy
importlib.reload(dummy)
