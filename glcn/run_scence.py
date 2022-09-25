import sys
import os
import copy
import json
import datetime

opt = dict()

#opt['dataset'] = "scence"
opt['losslr1'] = 0.0001
opt['decay_lr'] = 1.0

def generate_command(opt):
    cmd = 'python train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

laList = [0.1, 1, 10, 50, 100, 500, 1000, 10000, 50000]
#     #
laList1 = [0.01, 0.1, 1, 50, 100, 500]

sname = ("scence1", "scence2", "scence3", "scence4", "scence5", "scence6", "scence7", "scence8", "scence9", "scence10" )

for dname in sname:
    for lamb1 in laList1:
        for lamb in laList:
            opt['dataset'] = dname
            opt['lamb'] = lamb
            opt['lamb1'] = lamb1
            run(opt)
