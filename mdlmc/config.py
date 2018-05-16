from collections import defaultdict
from abc import ABCMeta
import importlib
import inspect
import pkgutil
import configparser

import mdlmc


def collect_classes(module):
    content = [getattr(module, name) for name in dir(module)]
    configurable_classes = [x for x in content if hasattr(x, "__show_in_config__")]
    return configurable_classes


def is_abstract(cls):
    return cls.__base__ is object and isinstance(cls, ABCMeta)


def top_class(cls):
    return cls.mro()[-2]


def has_parent_class(cls):
    return cls.mro()[-2] is not cls


def discover(mod):
    discoverable = defaultdict(dict)
    for importer, modname, ispkg in pkgutil.walk_packages(path=mod.__path__,
                                                          prefix=mod.__name__ + ".",
                                                          onerror=lambda x: None):
        print(modname)
        if not ispkg:
            module = importlib.import_module(modname)
            configurable_classes = collect_classes(module)
            if configurable_classes:
                for cls in configurable_classes:
                    parameters = inspect.signature(cls).parameters
                    exclude_from_config = getattr(cls, "__no_config_parameter__", [])
                    if is_abstract(cls):
                        continue
                    section_name = top_class(cls).__name__
                    discoverable[section_name]["parameters"] = \
                        {p.name: p.default if p.default is not inspect._empty else "EMPTY"
                         for p in parameters.values() if p.name not in exclude_from_config}
                    discoverable[section_name]["help"] = cls.__doc__
                    if has_parent_class(cls):
                        type_list = discoverable[section_name].setdefault("type", set())
                        type_list.add(cls)
            discoverable.update()
    #print(yaml.dump(dict(discoverable), default_flow_style=False))
    cp = configparser.ConfigParser(allow_no_value=True)
    for section in discoverable:
        cp.add_section(section)
        if discoverable[section].get("help", None):
            cp.set(section, "# " + discoverable[section]["help"])

        for param in discoverable[section]:
            if param == "parameters":
                for pm in discoverable[section][param]:
                    cp.set(section, pm, str(discoverable[section][param][pm]))
            elif param == "type":
                type_list = ", ".join([cls.__name__ for cls in discoverable[section][param]])
                cp.set(section, param, f"EMPTY  # Choose between {type_list}")
            elif param == "help":
                continue
            else:
                cp.set(section, param, str(discoverable[section][param]))

    from io import StringIO
    f = StringIO()
    cp.write(f)
    f.seek(0)
    print(f.read())


discover(mdlmc)