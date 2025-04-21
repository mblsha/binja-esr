# import this file before trying to import anything from binaryninja,
# as it will add the Binary Ninja Python API to the path
#
# This is useful for running unit tests from the command line.
import os
import sys

binjaroot_path = os.path.expanduser('~/Applications/Binary Ninja.app/Contents/Resources/python/')
if os.path.exists(binjaroot_path) and binjaroot_path not in sys.path:
    sys.path.append(binjaroot_path)

