import numpy as np, glob
for f in sorted(glob.glob("/data/ZAHA/preprocessed/train/*/coord.npy"))[:3]:
    c = np.load(f)
    print(f.split("/")[-2], "min:", c.min(0), "max:", c.max(0), "range:", c.max(0)-c.min(0))