#!/bin/bash

# python mswml/train_pl.py params/baseline-radam.py
# python mswml/train_pl.py params/test.py
# python mswml/train_pl.py params/baseline-radam-2.py           # <---- this works, really slow, but it works!
# python mswml/train_pl.py params/baseline-radam-onecycle.py
# python mswml/train_pl.py params/baseline-radam-steplr.py

# Karol trying to help - based on python mswml/train_pl.py params/baseline-radam-2.py
# python mswml/train_pl.py params/baseline-radam-2-km.py
# python mswml/train_pl.py params/baseline-radam-2-km-1.py

# python mswml/train_pl.py params/profiler.py
# python mswml/train_pl.py params/profiler-2.py
# python mswml/train_pl.py params/profiler-3.py

# python mswml/train_pl.py params/baseline-radam-3.py
# python mswml/train_pl.py params/baseline-radam-onecycle.py
# python mswml/train_pl.py params/baseline-radam-steplr.py
# python mswml/train_pl.py params/baseline-radam-onecycle-2.py
# python mswml/train_pl.py params/baseline-radam-onecycle-3.py
# python mswml/train_pl.py params/baseline-radam-onecycle-4.py
# python mswml/train_pl.py params/xunet-1clr-0.py

# python mswml/train_pl.py params/baseline-radam-onecycle-5.py


##### major changes in params py
# python mswml/train_pl.py params/xunet-1clr-0-mk.py
# python mswml/train_pl.py params/xunet-1clr-1-mk.py
# python mswml/train_pl.py params/xunet-1clr-2-mk.py
# python mswml/train_pl.py params/xunet-1clr-3.py
# python mswml/train_pl.py params/xunet-1clr-4.py

#### log images
# python mswml/train_pl.py params/test-2.py
# python mswml/train_pl.py params/xunet-1clr-5.py
# python mswml/train_pl.py params/xunet-1clr-6.py

#### add 3rd layer
# python mswml/train_pl.py params/xunet-1clr-7.py  # failed succesfully

# 4layers
# python mswml/train_pl.py params/xunet-1clr-8.py  # changed input to 64³  --> first working, after disabling wght std
# python mswml/train_pl.py params/xunet-1clr-9.py  # shorter training, larger LR
# python mswml/train_pl.py params/xunet-1clr-10.py  # larger accu; more data aug; focal * 5
# python mswml/train_pl.py params/xunet-1clr-11.py  # less accu
# python mswml/train_pl.py params/xunet-1clr-12.py  # even less accu; more epochs  --> FAILED - too high LR/too low accu
# python mswml/train_pl.py params/xunet-1clr-13.py # lower LR  -> not used, because 12 was ok
# python mswml/train_pl.py params/xunet-1clr-12.py  # even less accu; more epochs; once more, now with gradient clipping at 0.5
# python mswml/train_pl.py params/xunet-1clr-14.py # 12 with greater lr (and better viz)   --> TOO HIGH LR

# python mswml/train_pl.py params/xunet-1clr-15.py # class weights 1, 100
# python mswml/train_pl.py params/xunet-1clr-16.py # smoothing 0.1
# python mswml/train_pl.py params/xunet-1clr-17.py # class weights 100, 1


### High Performance training ™

# python mswml/train_pl.py params/xunet-hp-0.py # based on params/xunet-1clr-12.py
# python mswml/train_pl.py params/xunet-hp-1.py # precision bf16 --> not supported on RTX 8000
# python mswml/train_pl.py params/xunet-hp-2.py # larger accu, larger LR
# python mswml/train_pl.py params/xunet-hp-3.py # smaller accu (more updates!) smaller LR
# python mswml/train_pl.py params/xunet-hp-4.py # same as 3, but with changed start and final LR
# python mswml/train_pl.py params/xunet-hp-5.py # same as 3, but with crop foreground
# python mswml/train_pl.py params/xunet-hp-6.py # same as 4, but larger LR  --> FAILED with grad clip 0.5
# python mswml/train_pl.py params/xunet-hp-7.py # same as 4, but 10x longer (pct adjusted)
# python mswml/train_pl.py params/xunet-hp-6.py # same as 4, but larger LR; grad clip @ 0.1
# python mswml/train_pl.py params/xunet-hp-8.py # same as 6, but precision 32; grad clip @ 0.1
# python mswml/train_pl.py params/xunet-hp-9.py # still trying
# python mswml/train_pl.py params/xunet-hp-10.py # larger accu
# python mswml/train_pl.py params/xunet-hp-11.py # workers>0 (!)


# hp-4 is the best for now; long version: hp-7
# python mswml/train_pl.py params/xunet-hp-12.py # same as 4, workers=8
# python mswml/train_pl.py params/xunet-hp-13.py # same as 4, workers=20

# python mswml/train_pl.py params/xunet-hp-14.py # workers=20, larger model
# python mswml/train_pl.py params/xunet-hp-15.py # longer, lower LR



########### AUG
# python mswml/train_pl.py params/xunet-aug-base.py
# python mswml/train_pl.py params/xunet-aug-RandCoarseDropoutd.py
# python mswml/train_pl.py params/xunet-aug-RandCoarseDropoutd-1.py
# python mswml/train_pl.py params/xunet-aug-SavitzkyGolaySmoothd.py
#python mswml/train_pl.py params/xunet-aug-RandAdjustContrastd.py
#python mswml/train_pl.py params/xunet-aug-RandGibbsNoised.py
#python mswml/train_pl.py params/xunet-aug-RandBiasFieldd.py
#python mswml/train_pl.py params/xunet-aug-Rand3DElasticd.py
#python mswml/train_pl.py params/xunet-aug-RandCoarseDropoutd-2.py
#########################python mswml/train_pl.py params/xunet-aug-RandGridDistortion.py



########### Loss
# python mswml/train_pl.py params/xunet-loss-base.py
# python mswml/train_pl.py params/xunet-loss-dice.py
# python mswml/train_pl.py params/xunet-loss-focal.py
# python mswml/train_pl.py params/xunet-loss-ce.py
# python mswml/train_pl.py params/xunet-loss-bce.py
# python mswml/train_pl.py params/xunet-loss-gen-dice-focal.py
# python mswml/train_pl.py params/xunet-loss-ndsc.py
# python mswml/train_pl.py params/xunet-loss-tversky.py
# python mswml/train_pl.py params/xunet-loss-log-cosh.py
# python mswml/train_pl.py params/xunet-loss-ndsc-lr.py
# python mswml/train_pl.py params/xunet-loss-focal-lr.py
# python mswml/train_pl.py params/xunet-loss-ce-lr.py

# combine best losses
# python mswml/train_pl.py params/xunet-loss-ndsc-focal.py  # ndsc loss gives values approx 1 all the time -> multiply it
# python mswml/train_pl.py params/xunet-loss-ndsc-focal-ce.py # this equals to changing weight of CE... but let's try it


# input size
# python mswml/train_pl.py params/xunet-size-32.py
# python mswml/train_pl.py params/xunet-size-base.py


# model
# python mswml/train_pl.py params/model-unet.py
# python mswml/train_pl.py params/model-base.py
# python mswml/train_pl.py params/model-unet-1.py # adjust lr
# python mswml/train_pl.py params/baseline-radam-onecycle-6.py # compare older config
# python mswml/train_pl.py params/model-base-1.py # same intensity transforms as unet
# python mswml/train_pl.py params/model-unet-2.py # dont clip grad


# stacked data augmentation
# python mswml/train_pl.py params/xunet-aug-stack.py # based on xunet-aug-base
# python mswml/train_pl.py params/xunet-aug-stack-1.py # scale intensity at the beginning
# python mswml/train_pl.py params/xunet-aug-stack-2.py # rand spatial crop to the target shape
# python mswml/train_pl.py params/xunet-aug-stack-3.py # padding mode zeros
# python mswml/train_pl.py params/xunet-aug-stack-4.py # padding mode border
# python mswml/train_pl.py params/xunet-aug-stack-5.py # params as in article


# input size - once again
# python mswml/train_pl.py params/xunet-size-base-1.py
python mswml/train_pl.py params/xunet-size-128-1.py
