from collections import defaultdict
import importlib
import inspect
import pkgutil
import yaml

import mdlmc


def discover(mod):
    discoverable = defaultdict(dict)
    for importer, modname, ispkg in pkgutil.walk_packages(path=mod.__path__,
                                                          prefix=mod.__name__ + ".",
                                                          onerror=lambda x: None):
        print(modname)
        if not ispkg:
            module = importlib.import_module(modname)
            content = [getattr(module, name) for name in dir(module)]
            classes = [x for x in content if hasattr(x, "__show_in_config__")]
            if classes:
                for cls in classes:
                    parameters = inspect.signature(cls).parameters
                    discoverable[cls.__name__]["parameters"] = \
                        {p.name: p.default if p.default is not inspect._empty else "EMPTY"
                         for p in parameters.values() if p.name not in getattr(cls, "__no_config_parameter__", [])}
                    discoverable[cls.__name__]["help"] = cls.__doc__
            discoverable.update()
    print(yaml.dump(dict(discoverable), default_flow_style=False))



discover(mdlmc)