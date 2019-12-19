import os
from configparser import ConfigParser
import platform
import logging

this_dir = os.path.abspath(os.path.dirname(__file__))
if platform.node() == "landons-laptop":
    settings_file = os.path.join(this_dir, "settings.ini")
else:
    if platform.system() == "Windows":
        settings_file = os.path.join(os.path.expanduser("~"), "AppData", "Local", "colocalizer", "settings.ini")
    else:
        settings_file = os.path.join(os.path.expanduser("~"), ".colocalizer", "settings.ini")

logger = logging.getLogger("colocalizer.setup")

def copy_settings_default():
    config = ConfigParser()
    config['DEFAULT'] = {"last_dir":os.path.expanduser("~")}
    config['DYES'] = {"red":"cy3, tv2-t2-sr",
                      "green":"fitc, tv2-t1-sr",
                      "blue":"dapi, hoechst, tv2-t3-sr"}
    config['THRESHOLDS'] = {"red":"4000","green":"4000","blue":"2000"}
    config['SIZES'] = {"red": "5", "green": "5", "blue": "2000"}
    if not os.path.exists(settings_file):
        logger.debug("Creating new settings file")
        if not os.path.exists(os.path.dirname(settings_file)):
            os.makedirs(os.path.dirname(settings_file))
        with open(settings_file, 'w') as configfile:
            config.write(configfile)