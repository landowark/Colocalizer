import logging.handlers
from ui import Ui_MainWindow
from PyQt5 import QtWidgets
import javabridge
import os
import bioformats as bf
import logging
import platform

log_config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', "log4j.properties")

def setup_logging():
    global logger
    logger = logging.getLogger("lamin_quant")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    this_dir = os.path.abspath(os.path.dirname(__file__))
    if platform.node() == "landons-laptop":
        log_path = os.path.join(os.path.dirname(this_dir), "logs", "LaminQuant.log")
    else:
        log_path = os.path.join(os.path.expanduser("~"), "AppData", "Local", "LaminQuant", "LaminQuant.log")
    if not os.path.exists(os.path.dirname(log_path)):
        try:
            os.makedirs(os.path.dirname(log_path))
        except:
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LaminQuant.log')
    fh = logging.handlers.RotatingFileHandler(log_path, mode='a',
                                              maxBytes=100000, backupCount=3, encoding=None, delay=False)
    fh.setLevel(logging.DEBUG)
    fh.name = "File"
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.name = "Stream"
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(ch_formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.debug("Starting up.")
    logger.debug((f"Using {log_path}"))



if __name__ == "__main__":
    import sys
    setup_logging()
    javabridge.start_vm(
        args=[
            "-Dlog4j.configuration=file:{}".format(log_config),
        ],
        class_path=bf.JARS,
        run_headless=True
    )
    try:
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
    finally:
        javabridge.kill_vm()
