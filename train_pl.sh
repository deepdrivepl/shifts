#!/bin/bash

# python mswml/train_pl.py params/baseline-radam.py
# python mswml/train_pl.py params/test.py
# python mswml/train_pl.py params/baseline-radam-2.py           # <---- this works, really slow, but it works!
# python mswml/train_pl.py params/baseline-radam-onecycle.py
# python mswml/train_pl.py params/baseline-radam-steplr.py

# Karol trying to help - based on python mswml/train_pl.py params/baseline-radam-2.py
python mswml/train_pl.py params/baseline-radam-2-km.py

