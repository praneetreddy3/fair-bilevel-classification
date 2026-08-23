"""
Driver to generate client splits for adult/credit using generate_datasets.py's
split_existing_dataset(), with a comparable 5-client equal-split setup to our
own pipeline's final runs (--num_clients 5). Not part of FairSynData's method
code -- just invokes the existing splitting utility for the new datasets.
"""
import sys
import shutil
import os

import generate_datasets as gd

DATASET = sys.argv[1]  # 'adult' or 'credit'

gd.dataset_name = DATASET
gd.number_of_clients = 5
gd.client_dr_ratio = [0.2, 0.2, 0.2, 0.2, 0.2]

client_train, client_test = gd.split_existing_dataset(
    dataset_name=DATASET, ratio_given=False, fair_ratio=False, alpha=None, number_of_clients=5
)

print(f"base_dir = {gd.base_dir}")
print(f"num train clients = {len(client_train)}, num test clients = {len(client_test)}")

# Copy into datasets/dummy_run/ with the naming pattern main.py expects
dummy_dir = os.path.join("datasets", "dummy_run")
os.makedirs(dummy_dir, exist_ok=True)
for f in os.listdir(gd.base_dir):
    if f.startswith(f"{DATASET}_") and f.endswith(".csv"):
        # e.g. adult_20260814_train_3.csv -> extract train/test + client id
        parts = f.split("_")
        kind = "train" if "train" in f else ("test" if "test" in f else None)
        if kind is None:
            continue
        client_id = f.split("_")[-1].split(".")[0]
        dest = os.path.join(dummy_dir, f"{DATASET}_dummy_{kind}_{client_id}.csv")
        shutil.copy(os.path.join(gd.base_dir, f), dest)
        print(f"copied {f} -> {dest}")
