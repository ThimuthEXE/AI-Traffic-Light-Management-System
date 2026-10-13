#!/usr/bin/env python
"""
AITraffic Platform Master Entry Point
Run without arguments for interactive console, or with subcommands:
    python main.py sim 4
    python main.py dashboard
    python main.py test
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from aitraffic.cli import main

if __name__ == "__main__":
    main()
