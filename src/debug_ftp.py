


######### this is not fixable, the two attacks are identical in everything except packet rate
#  and that distribution overlaps too much to be reliable anyways, a cic-ids2018 flaw
#  soit we merge the two classes under one label DoS/BruteForce-Port21 in preprocess negh we just accept it 

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

raw_df = pd.read_pickle(PROCESSED_DIR / "test_raw.pkl")
le     = pickle.load(open(PROCESSED_DIR / "label_encoder.pkl", "rb"))

ftp_id  = list(le.classes_).index("FTP-BruteForce")
slow_id = list(le.classes_).index("DoS attacks-SlowHTTPTest")

ftp_rows  = raw_df[raw_df["label_enc"] == ftp_id]
slow_rows = raw_df[raw_df["label_enc"] == slow_id]

feature_list = json.loads((BASE_DIR / "data" / "feature_list.json").read_text())
import json
feature_list = json.loads((BASE_DIR / "data" / "feature_list.json").read_text())

# Compute mean ratio for every feature — highest ratio = most discriminative
ftp_means  = ftp_rows[feature_list].mean()
slow_means = slow_rows[feature_list].mean()

ratio = (ftp_means / (slow_means.replace(0, np.nan))).abs()
ratio.name = "ftp/slow mean ratio"

comparison = pd.DataFrame({
    "ftp_mean" : ftp_means,
    "slow_mean": slow_means,
    "ratio"    : ratio,
}).sort_values("ratio", ascending=False)

print("=== Top 15 most discriminative features (FTP vs SlowHTTPTest) ===")
print(comparison.head(15).to_string())
print("\n=== Bottom 10 (least discriminative) ===")
print(comparison.tail(10).to_string())