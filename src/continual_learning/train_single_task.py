import os
import re
import argparse
import torch
from typing import Any, Callable, Dict, List, Optional

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    modeling_utils,
)

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])
import sys
import_dir = os.path.join(ROOT_DIR, 'src')
sys.path.insert(0, import_dir)

from metrics import compute_accuracy
from model.utils import render_chat
from tasks.AGNEWS import (
    example_to_messages as agnews_example_to_messages
)