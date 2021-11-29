# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:55:33 2021

@author: basti
"""

import numpy as np

nMax = 100
n1 = 30
n2 = 45

l1 = np.unique(np.random.randint(0, nMax, n1))
l2 = np.unique(np.random.randint(0, nMax, n2))

d = set(l1).intersection(l2)

print(len(l1), len(l2), len(d))