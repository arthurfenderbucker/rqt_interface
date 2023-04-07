#!/usr/bin/env python

import sys

from rqt_interface.module import MyPlugin
from rqt_gui.main import Main

plugin = 'rqt_interface'
main = Main(filename=plugin)
sys.exit(main.main(standalone=plugin))