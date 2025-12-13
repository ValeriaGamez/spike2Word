from pathlib import Path
import re
import numpy as np
import scipy.io as sio

# --- paths ---
sessionPath = "/Users/alp/Desktop/USC FALL 2025/Project/projectData/train/t12.2022.04.28_first5.mat"
txt_dir     = Path("/Users/alp/desktop/usc fall 2025/Project/findings/128-input/electrodes_3_neuron")

# --- load .mat ---
dat = sio.loadmat(sessionPath, squeeze_me=True, struct_as_record=False)
tx1 = dat["tx1"]  # tx1[i] is (T, 256)

# --- patch each txt ---
for txt_path in sorted(txt_dir.glob("electrodes_sentence_*.txt")):
    # extract trailing index i from filename (e.g., electrodes_sentence_4.txt -> i=4)
    m = re.search(r"(\d+)$", txt_path.stem)
    if not m:
        raise ValueError(f"Can't parse index from filename: {txt_path.name}")
    i = int(m.group(1))

    X = np.loadtxt(txt_path)            # (T, 256) from txt
    mat = np.asarray(tx1[i])            # (T, 256) from .mat

    # sanity checks
    if X.shape[0] != mat.shape[0]:
        raise ValueError(f"{txt_path.name}: row mismatch txt={X.shape[0]} vs tx1[{i}]={mat.shape[0]}")
    if X.shape[1] < 128:
        raise ValueError(f"{txt_path.name}: expected >= 128 columns, got {X.shape[1]}")
    if mat.shape[1] < 64:
        raise ValueError(f"tx1[{i}]: expected >= 64 columns, got {mat.shape[1]}")

    # replace TXT cols 64..127 with MAT cols 0..63
    X[:, 64:128] = mat[:, :64]

    out_path = txt_path.with_name(txt_path.stem + "_patched.txt")  # change to txt_path to overwrite
    np.savetxt(out_path, X, delimiter="\t", fmt="%.6g")
    print(f"[ok] {txt_path.name} -> {out_path.name}")
