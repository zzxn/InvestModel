#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import logging
import os
from typing import List

import yaml

_logger = logging.getLogger("yaml_config")


class ConfigWarning(RuntimeWarning):
    pass


class ConfigError(RuntimeError):
    pass


def get_logger():
    return _logger


class Config:
    def __init__(self, config_yaml_path):
        get_logger().info(f'Reading config file {config_yaml_path}...')
        if not os.path.exists(config_yaml_path):
            raise ConfigError(f'Cannot find config.yaml at {config_yaml_path}.')
        with open(config_yaml_path, 'r') as f:
            cfg = f.read()
        self._cfg = yaml.load(cfg, yaml.FullLoader)
        get_logger().info('Read config successfully.')

    def __getitem__(self, key_str: str):
        self._check_key_str(key_str)

        keys: List[str] = key_str.split('.')
        item = self._get_item_recurse(keys)
        if isinstance(item, dict):
            import warnings
            message = 'Access config item which is not at the end point. Are you sure it is correct?'
            warnings.warn(message, ConfigWarning)
        item = copy.deepcopy(item)
        return item

    def __setitem__(self, key_str: str, value):
        self._check_key_str(key_str)

        keys: List[str] = key_str.split('.')
        if len(keys) == 1:
            parent = self._cfg
        else:
            parent = self._get_item_recurse(keys[:-1])
        if not isinstance(parent, dict):
            raise ConfigError('Invalid assignment.')
        if keys[-1] not in parent:
            raise ConfigError(f"{keys[-1]} is not in .{'.'.join(keys[:-1])}")
        if isinstance(parent[keys[-1]], dict):
            import warnings
            message = 'Access config item which is not at the end point. Are you sure it is correct?'
            warnings.warn(message, ConfigWarning)
        old_value = parent[keys[-1]]
        if (old_value is not None) and (not isinstance(value, type(old_value))):
            import warnings
            message = 'Old value and new value are not the same type'
            warnings.warn(message, ConfigWarning)
        parent[keys[-1]] = value
        return old_value

    def __call__(self, key_str: str, default=None):
        try:
            val = self[key_str]
        except ConfigError:
            val = None
        val = val or default
        return val

    def _check_key_str(self, key_str: str):
        if not isinstance(key_str, str):
            raise ConfigError('key_str must be str.')
        if not key_str:
            raise ConfigError('key_str cannot be empty.')

    def _get_item_recurse(self, keys: List[str]):
        item = self._cfg
        key_prefix = '[ROOT]'
        for key in keys:
            if (item is None) or (not isinstance(item, dict)) or (key not in item.keys()):
                raise ConfigError(f'Cannot find {key} in {key_prefix}')
            key_prefix += '.' + key
            item = item[key]
        return item

    def __str__(self):
        return yaml.dump(self._cfg)

    def __repr__(self):
        return str(self)
