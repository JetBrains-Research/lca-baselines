 ```python
import ctypes
import time
import pyrasite
import bcc

symbol = ""
implementation_addr = 0
resolver_addr = 0
implementation_offset = 0

def get_symbol():
    global symbol
    symbol = ctypes.cdll.LoadLibrary("libc.so.6").strlen
