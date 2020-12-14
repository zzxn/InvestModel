#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from config.yaml_config import Config

WORKING_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
CONFIG_YAML_PATH = os.path.join(WORKING_DIR, '..', 'config.yaml')

# init config
config = Config(CONFIG_YAML_PATH)
