from __future__ import absolute_import, division, print_function

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import random
import sys
from datetime import datetime

import pandas as pd
from utils.keras_utils import set_keras_growth
from utils.runutils import getArgs, runalldatasets
from utils.storage_utils import createdir

ROOT_DIR = "."


def writejson(filename, data):
    with open(filename, "w") as outfile:
        json.dump(data, outfile)


def main(
    datasets: str = "iris,use,eco,glass,heart,car,hay,mam,ttt,pim,bal,who,mon,cmc",
    prefix: str = "",
):
    sys.argv = [
        "runner.py",
        "--kfold=5",
        "--epochs=200",
        "--methods",
        "eSNN:rprop:200:split:0.15",
        "--datasets",
        datasets,
        "--onehot",
        "True",
        "--multigpu",
        "False",
        "--batchsize",
        "1000",
        "--hiddenlayers",
        "13,13",
        "--gpu",
        "0",
        f"--prefix={prefix}",
        "--n",
        "5",
        "--cvsummary",
        "False",
        "--printcv",
        "False",
        "--seed",
        "42",
    ]
    run()


def run():
    args = getArgs()

    if args.seed == None:
        seed = random.randrange(sys.maxsize)
        args.seed = seed
        print(f"generating new random seed: {seed}")
    else:
        print(f"setting random seed to: {args.seed}")

    random.seed(args.seed)

    datasetlist = args.datasets

    print(f"doing experiment with {datasetlist} in that order")

    runlist = args.methods

    set_keras_growth(args.gpu)

    prefix = f"{ROOT_DIR}/output/runner"
    if args.prefix is not None:
        prefix = args.prefix

    rootpath = prefix + datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    createdir(rootpath)

    writejson(f"{rootpath}/settings.json", sys.argv[1:])

    if args.callbacks is not None:
        callbacks = args.callbacks
    else:
        callbacks = list()
    alphalist = [0.8]
    nresults = list()

    start_time = datetime.now()
    print(f"Starting experiments at: {start_time}")

    for i in range(0, args.n):
        run_start_time = datetime.now()
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
        run_end_time = datetime.now()
        print(f"Run {i+1}/{args.n} completed in: {run_end_time - run_start_time}")
        writejson(f"{rootpath}/data.json", nresults)

    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"All experiments completed at: {end_time}")
    print(f"Total duration: {total_duration}")

    resdf = pd.DataFrame(nresults)
    resdf.to_csv(
        f"{rootpath}/results_{args.kfold}_fold_{args.epochs}_epochs{'_onehot' if args.onehot else ''}.csv"
    )


if __name__ == "__main__":
    run()
