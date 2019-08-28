#!/bin/python

import numpy as np

arr = []

for i in range(10):
    arr.append(i)

for el in arr:
    print("Original: {}".format(el))

arr = [ el * 2 for el in arr ]

for el in arr:
    print("Modified: {}".format(el))

print(arr[slice(0, 5, 2)])

map = { el : el * 5 for el in arr}

print(map)

for key, value in map.items():
    print("{}: {}".format(key, value))


map = {
        "red": 6,
        "green": 7,
        "violet": 8,
        "red": 9,
        }

for i, (k, v) in enumerate(map.items()):
    print("{}. {}: {}".format(i, k, v))

nth = { i: v for i, (k, v) in enumerate(map.items())}

print(nth)
