import os
import sys

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(CODE_DIR))
