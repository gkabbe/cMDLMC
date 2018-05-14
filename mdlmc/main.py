import argparse
import logging.config
import pathlib
import yaml


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("configfile", help="Text file containing the configuration for the cMD/LMC"
                                           "scheme")
    args = parser.parse_args()
    # Check if logging config file exists in current directory
    logfile_path = pathlib.Path("logging.cfg")
    if not logfile_path.exists():
        logfile_path = pathlib.Path(__file__).parents[1] / "logging.yaml"
    with open(logfile_path, "r") as f:
        logging_dict = yaml.load(f)
    print(logging_dict)
    logging.config.dictConfig(logging_dict)
    logger = logging.getLogger(__name__)

    with open(args.configfile, "r") as f:
        options = yaml.load(f)

    logger.debug(options)





