#!/usr/bin/env python3


# ME499-S20 Python Lab 7
# Programmer: Jacob Gray
# Last Edit: 5/27/2020


import itertools as it
import numpy as np


# product(A, repeat=4) == product(A, A, A, A)


num_links = 3  # Number of links
interval = 4  # Number of intervals to iterate over range

counter = 0



print(counter)
print(it.product(np.linspace(0, 2 * np.pi, interval, endpoint=False), repeat=num_links))
