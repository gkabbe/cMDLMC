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


def get_parameters(cls):
    parameters = inspect.signature(cls).parameters
    exclude_from_config = getattr(cls, "__no_config_parameter__", [])
    param_dict = {p.name: p.default if p.default is not inspect._empty else "EMPTY"
                  for p in parameters.values() if p.name not in exclude_from_config}
    return param_dict


def get_unique_parameters(cls):
    top_cl = top_class(cls)
    related_classes = {top_cl, *top_cl.__subclasses__()}.difference({cls})
    print("related to", cls, ":", related_classes)
    params = set(get_parameters(cls).keys())
    for c in related_classes:
        c_params = set(get_parameters(c).keys())
        params -= c_params
    return params


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
                    if is_abstract(cls):
                        continue
                    section_name = top_class(cls).__name__
                    # Check if there exist inherited classes
                    if top_class(cls).__subclasses__():
                        type_list = discoverable[section_name].setdefault("type", set())
                        type_list.add(cls)
                        unique_params = get_unique_parameters(cls)
                    else:
                        unique_params = {}
                    cls_name = cls.__name__
                    param_dict = get_parameters(cls)
                    if unique_params:
                        for k, v in param_dict.items():
                            if k in unique_params:
                                param_dict[k] = f"{v}  # {cls_name}"
                    section_params = discoverable[section_name].setdefault("parameters", {})
                    section_params.update(param_dict)
                    if cls.__doc__:
                        discoverable[section_name]["help"] = cls.__doc__.replace("\n", "\n#")
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