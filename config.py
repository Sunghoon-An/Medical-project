#!/usr/bin/env python
# coding: utf-8

import os

BASE = '.'
DATA = '/data/gruads/backup/'
AGGREGATE = '/data/gruads/preprocess/aggregate'

# aggregation normal
RESULT = os.path.join(BASE,'result')
LOG = os.path.join(BASE,'logs')
BACKUP = os.path.join(BASE,'backup')

FEATURE_SIZE = 10 # embbeding feature size
# MAX_LEN = 40