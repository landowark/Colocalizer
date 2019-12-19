import os
from configparser import ConfigParser
import platform
import logging

this_dir = os.path.abspath(os.path.dirname(__file__))
if platform.node() == "landons-laptop":
    settings_file = os.path.join(this_dir, "settings.ini")
else:
    settings_file = os.path.join(os.path.expanduser("~"), "AppData", "Local", "LaminQuant", "settings.ini")

logger = logging.getLogger("lamin_quant.setup")

def copy_settings_default():
    config = ConfigParser()
    config['DEFAULT'] = {"last_dir":os.path.expanduser("~")}
    config['DYES'] = {"blue":"dapi, hoechst, tv2-t3-sr",
                      "red":"cy3, tv2-t2-sr",
                      "green":"fitc, tv2-t1-sr"}
    config['THRESHOLDS'] = {"Blue":"2000", "Green":"4000"}
    if not os.path.exists(settings_file):
        logger.debug("Creating new settings file")
        if not os.path.exists(os.path.dirname(settings_file)):
            os.makedirs(os.path.dirname(settings_file))
        with open(settings_file, 'w') as configfile:
            config.write(configfile)