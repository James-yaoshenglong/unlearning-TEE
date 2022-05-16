import numpy as np
import json
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--shards",
    default=5,
    type=int,
    help="Split the dataset in the given number of shards in an optimized manner (PLS-GAP partitionning) according to the given distribution, create the corresponding splitfile",
)
parser.add_argument(
    "--requests",
    default=1,
    type=int,
    help="Generate the given number of unlearning requests according to the given distribution and apply them directly to the splitfile",
)
parser.add_argument(
    "--distribution",
    default="uniform",
    help="Assumed distribution when used with --shards, sampling distribution when used with --requests. Use 'reset' to reset requestfile, default uniform",
)
parser.add_argument("--container", default="default", help="Name of the container")
parser.add_argument(
    "--dataset",
    default="../datasets/purchase/datasetfile",
    help="Location of the datasetfile, default ../datasets/purchase/datasetfile",
)
parser.add_argument("--label", default="latest", help="Label, default latest")
args = parser.parse_args()

# Load dataset metadata.
with open(args.dataset) as f:
    datasetfile = json.loads(f.read())

if args.shards != None:
    # If distribution is uniform, split without optimizing.
    if args.distribution == "uniform":
        partition = np.split(
            np.arange(0, datasetfile["nb_train"]),
            [
                t * (datasetfile["nb_train"] // args.shards)
                for t in range(1, args.shards)
            ],
        )
        np.save("../containers/{}/splitfile.npy".format(args.container), partition)
        # requests = np.array([[] for _ in range(args.shards)])
        # np.save(
        #     "containers/{}/requestfile:{}.npy".format(args.container, args.label),
        #     requests,
        # )