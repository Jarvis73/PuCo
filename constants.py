import os
from pathlib import Path


if 'RUN_POSITION' in os.environ and os.environ['RUN_POSITION'] == 'paddlecloud':
    on_cloud = True
else:
    on_cloud = False
