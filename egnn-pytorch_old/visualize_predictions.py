#!/usr/bin/env python3
"""
Visualize EGNN predictions from checkpoint directory.
Shows side-by-side comparison of true vs predicted mesh at a specific timestep.

This is a thin wrapper entrypoint that calls egnn_visualize.main().
All visualization logic has been moved to egnn_visualize.py for better modularity.
"""

import sys
import os

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from egnn_visualize import main

if __name__ == '__main__':
    main()
