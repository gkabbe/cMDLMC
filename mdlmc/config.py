# coding=utf-8

import configparser
import importlib
import inspect
import pkgutil
import warnings
from abc import ABCMeta
from collections import defaultdict
from io import StringIO

import mdlmc


def show_in_config(x):
    return getattr(x, "__show_in_config__", False)


def collect_classes(module):
    content = [getattr(module, name) for name in dir(module)]
    configurable_classes = [x for x in content if show_in_config(x)]
    return configurable_classes


def is_abstract(cls):
    return cls.__base__ is object and isinstance(cls, ABCMeta)


def top_class(cls):
    return cls.mro()[-2]


def has_parent_class(cls):
    return cls.mro()[-2] is not cls


def get_parameters(cls):
    what_to_inspect = getattr(cls, "__show_signature_of__", None)
    inspectable = getattr(cls, what_to_inspect) if what_to_inspect else cls
    parameters = inspect.signature(inspectable).parameters
    exclude_from_config = getattr(cls, "__no_config_parameter__", [])
    param_dict = {p.name: p.default if p.default is not inspect._empty else "EMPTY"
                  for p in parameters.values() if p.name not in exclude_from_config}
    annotation_dict = {}
    for name in param_dict:
        annotation = parameters[name].annotation
        annotation_dict[name] = getattr(annotation, "__name__", annotation)

    return param_dict, annotation_dict


def get_unique_parameters(cls):
    top_cl = top_class(cls)
    related_classes = {top_cl, *top_cl.__subclasses__()}.difference({cls})
    params = set(get_parameters(cls)[0].keys())
    for c in related_classes:
        c_params = set(get_parameters(c)[0].keys())
        params -= c_params
    return params


def discover(mod):
    discoverable = defaultdict(dict)
    for importer, modname, ispkg in pkgutil.walk_packages(path=mod.__path__,
                                                          prefix=mod.__name__ + ".",
                                                          onerror=lambda x: None):
        if not ispkg:
            module = importlib.import_module(modname)
            configurable_classes = collect_classes(module)
            if configurable_classes:
                for cls in configurable_classes:
                    section_name = top_class(cls).__name__
                    section_help = ""
                    if top_class(cls).__doc__:
                        section_help += top_class(cls).__doc__.replace("\n", "\n#")
                    if top_class(cls).__init__.__doc__:
                        section_help += top_class(cls).__init__.__doc__.replace("\n", "\n#")
                    discoverable[section_name]["help"] = section_help

                    if is_abstract(cls):
                        continue
                    # Check if there exist inherited classes
                    if top_class(cls).__subclasses__():
                        type_list = discoverable[section_name].setdefault("type", set())
                        type_list.add(cls)
                        unique_params = get_unique_parameters(cls)
                    else:
                        unique_params = {}
                    cls_name = cls.__name__
                    param_dict, annotation_dict = get_parameters(cls)
                    cls_info = {}
                    if unique_params:
                        for k, v in param_dict.items():
                            if k in unique_params:
                                cls_info[k] = f"(only {cls_name})"
                    for k in param_dict.keys():
                        param_dict[k] = f"{param_dict[k]:}  #  type {annotation_dict.get(k, 'None')} {cls_info.get(k, '')}"
                    section_params = discoverable[section_name].setdefault("parameters", {})
                    section_params.update(param_dict)
            discoverable.update()

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

    f = StringIO()
    cp.write(f)
    f.seek(0)
    cfg_str = f.read()
    for line in cfg_str.splitlines():
        try:
            left, right = line.split("#")
        except ValueError:
            print(line)
        else:
            if "=" in left:
                print(f"{left:40}  # {right}")
            else:
                print("#", right.strip())


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        discover(mdlmc)


if __name__ == "__main__":
    main()
