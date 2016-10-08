# -*- coding: utf-8 -*-
# 默认不输出 log

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())
