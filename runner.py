from __future__ import absolute_import, division, print_function

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from models.model_utils import makeANNModel
from utils.keras_utils import set_keras_growth
from utils.storage_utils import createdir
from datetime import datetime

from scikeras.wrappers import KerasClassifier
from utils.runutils import runalldatasets, getArgs
import sys
import numpy as np
import pandas as pd
import json
import random


# temporary remove "use" dataset because of openpyxl bug can't read xls
def test():
    sys.argv = [
        "runner.py",
        "--kfold=5",
        "--epochs=200",
        "--methods",
        "eSNN:adam:200:split:0.15,chopra:adam:200:gabel,gabel:adam:200:gabel,t3i1:adam:200:split,t1i1,t2i1",
        "--datasets",
        "iris,eco,glass,heart,car,hay,mam,ttt,pim,bal,who,mon,cmc",
        "--onehot",
        "True",
        "--multigpu",
        "False",
        "--batchsize",
        "1000",
        "--hiddenlayers",
        "13,13",
        "--gpu",
        "0,1",
        "--prefix=200epochs",
        "--n",
        "5",
        "--cvsummary",
        "False",
        "--printcv",
        "False",
    ]
    main()


def main():
    args = getArgs()

    if args.seed is None:
        seed = random.randrange(sys.maxsize)
        args.seed = seed
        print(f"generating new random seed: {seed}")
    else:
        print(f"setting random seed to: {args.seed}")

    random.seed(args.seed)

    datasetlist = args.datasets

    print(f"doing experiment with {datasetlist} in that order")

    k = args.kfold
    results = {}
    runlist = args.methods

    set_keras_growth(args.gpu)

    prefix = "runner"
    if args.prefix is not None:
        prefix = args.prefix
    rootpath = prefix + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    createdir(rootpath)

    min_retain_losses = list()
    min_losses = list()

    writejson(f"{rootpath}/settings.json", sys.argv[1:])

    if args.callbacks is not None:
        callbacks = args.callbacks
    else:
        callbacks = list()
    alphalist = [0.8]
    nresults = list()
    for i in range(0, args.n):
        nresults.append(
            runalldatasets(
                args,
                callbacks,
                datasetlist,
                rootpath,
                runlist,
                alphalist=alphalist,
                n=i,
                printcvresults=args.cvsummary,
                printcv=args.printcv,
                doevaluation=args.doevaluation,
            )
        )
        writejson(f"{rootpath}/data.json", nresults)

    resdf = pd.DataFrame(nresults)
    resdf.to_csv(
        f"{rootpath}/results_{args.kfold}kfold_{args.epochs}epochs_{args.onehot}onehot.csv"
    )


def writejson(filename, data):
    with open(filename, "w") as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    if len(sys.argv) == 1:  # No arguments provided
        test()
    else:
        main()
