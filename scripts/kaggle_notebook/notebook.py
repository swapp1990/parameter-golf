#!/usr/bin/env python
"""Parameter Golf experiment runner on Kaggle T4 GPU."""
import os, subprocess, sys, time

# === Setup ===
print("=== Setup ===")
os.chdir("/kaggle/working")
if not os.path.exists("parameter-golf"):
    subprocess.run(["git", "clone", "https://github.com/swapp1990/parameter-golf.git"], check=True)
os.chdir("parameter-golf")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "zstandard"], check=True)

# Download data (val + 1 train shard)
if not os.path.exists("data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"):
    print("Downloading data...")
    subprocess.run([sys.executable, "data/cached_challenge_fineweb.py",
                    "--variant", "sp1024", "--train-shards", "1"], check=True)

# Check if GPU needs older PyTorch (P100 = sm_60, needs torch<2.6)
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 7:
        print(f"GPU sm_{cap[0]}{cap[1]} needs older PyTorch, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch==2.4.0", "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
        import importlib
        importlib.invalidate_caches()
        torch = importlib.import_module("torch")
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

# === Training: 300s smoke test ===
print("\n=== Training 300s smoke test ===")
env = os.environ.copy()
env.update({
    "RUN_ID": "kaggle_t4_smoke",
    "TRAIN_BATCH_TOKENS": "16384",
    "MAX_WALLCLOCK_SECONDS": "300",
    "WARMDOWN_ITERS": "400",
    "MLP_MULT": "3",
    "MUON_WD": "0.04",
    "NUM_LAYERS": "11",
    "TRAIN_SEQ_LEN": "2048",
    "MATRIX_LR": "0.04",
    "SCALAR_LR": "0.04",
    "VAL_LOSS_EVERY": "200",
})
t0 = time.time()
subprocess.run(["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"], env=env)
print(f"Training done in {time.time()-t0:.0f}s")

print("\n=== Done ===")
