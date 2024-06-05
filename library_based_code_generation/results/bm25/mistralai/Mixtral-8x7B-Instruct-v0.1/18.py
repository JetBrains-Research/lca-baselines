 ```python
import sys
import threading
import time
from functools import partial

import silx
from silx.gui import qt, widgets
from silx.gui.utils.concurrent import ThreadPoolButton
