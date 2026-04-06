# Parameter Golf Project

## CRITICAL RULES (READ EVERY TIME)

1. **NEVER block the conversation.** Any SSH command that takes >15s MUST use `run_in_background: true` on the Bash tool. This includes: training runs, pip install, git clone, data downloads, any heavy python script.
2. **NEVER chain multiple commands in one SSH call.** Run ONE command at a time.
3. **Poll, don't wait.** After launching a background command, poll with quick foreground `ssh ... "tail -5 /runpod-volume/<log>"` commands.
4. **ONE background task per training run.** No extra wait/poll loops as background tasks.
5. **Download checkpoints BEFORE stopping pods.** Save to `D:/MyProjects/Claude/parameter-golf/dashboard/checkpoints/<exp_name>/`.
6. **Always give the user a `tail -f` command after starting any background task.** Format: `! ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> "tail -f /runpod-volume/<log>"`

## Running Experiments on RunPod

### SSH Connection

```bash
# Get SSH info for a pod
runpodctl ssh info <pod-id>

# SSH command format (from ssh info output)
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> "<command>"
```

### Starting a Training Run

```bash
# MUST use run_in_background: true on the Bash tool
ssh ... "cd /runpod-volume/parameter-golf && nohup bash -c '\
  RUN_ID=<name> \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  TRAIN_BATCH_TOKENS=65536 \
  MAX_WALLCLOCK_SECONDS=300 \
  WARMDOWN_ITERS=400 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py \
' > /runpod-volume/<name>.log 2>&1 &"
```

- Always set `TRAIN_BATCH_TOKENS=65536` for single GPU (default 524288 is for 8xH100)
- Scale `WARMDOWN_ITERS` to ~30% of expected steps (not 3000 for short runs)

### Checking Progress

```bash
# These are fast (<5s) — run as normal foreground commands
ssh ... "tail -5 /runpod-volume/<name>.log"
ssh ... "grep val_bpb /runpod-volume/<name>.log"
ssh ... "nvidia-smi"
ssh ... "ps aux | grep train"
```

### Fast vs Background Commands

**Background (run_in_background: true):** training, pip install, git clone, data download, any python script
**Foreground (normal):** echo, ls, cat, tail, head, wc, ps, nvidia-smi, kill, mv, cp, mkdir, git checkout

## RunPod Pod Management

`runpodctl` is installed. API key is pre-configured. Use `MSYS_NO_PATHCONV=1` prefix to prevent Git Bash path mangling.

```bash
runpodctl pod list                    # list pods
runpodctl network-volume list         # list volumes
runpodctl ssh info <pod-id>           # SSH connection details

# Create 1xH100 SXM pod
MSYS_NO_PATHCONV=1 runpodctl pod create \
  --name "pod-name" \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --image "runpod/parameter-golf:latest" \
  --network-volume-id "y26gyuaocv" \
  --volume-mount-path "/runpod-volume" \
  --container-disk-in-gb 20 \
  --data-center-ids "US-MO-1" \
  --ports "22/tcp"

runpodctl pod stop <pod-id>           # stop (keeps data)
runpodctl pod start <pod-id>          # restart
runpodctl pod remove <pod-id>         # terminate
```

**Key resources:**
- Network volume: `param-golf-shared` (id: `y26gyuaocv`, 50GB, US-MO-1)
- Volume mount: `/runpod-volume/`
- Docker image: `runpod/parameter-golf:latest`
- GPU IDs: `NVIDIA H100 80GB HBM3` (SXM), `NVIDIA H100 PCIe`
- Always use network volumes. Always terminate old pods before creating new.

## Vast.ai GPU Selection

**Choose GPU based on the task, not just price.** A $0.07/hr GPU that can't run the experiment wastes more money than a $0.36/hr GPU that finishes in 30 min.

### GPU Tiers

| Task | Min GPU | Recommended | Why |
|------|---------|-------------|-----|
| **Training (300s smoke test)** | RTX 4090 (24GB) | RTX 4090 | Needs fast step time + enough VRAM for float32 model + grads |
| **Eval-only (no training)** | RTX 4060 Ti (16GB) | RTX 4060 Ti | Forward-only, no grads, smaller batches OK |
| **Full training run** | A100 / H100 | Use RunPod H100 | Need 80GB+ for 8x batch, fast matmul |

### Rules

1. **Training experiments need RTX 4090 or better.** Float32 model (106MB) + optimizer states + gradients + activations = 15-20GB. Cheaper GPUs OOM on backward pass.
2. **eval_val takes ~30 min on 2080 Ti, ~2 min on 4090, ~20s on H100.** If your experiment calls eval_val, factor this in.
3. **Never use RTX 2080 Ti / RTX 4060 Ti for training experiments.** They're eval-only GPUs for this model.
4. **Always use `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` image** — needed for torch.compile.
5. **Model MUST stay float32 after dequantization.** Never call `.bfloat16()` — it destroys int5/int6 precision (+0.65 BPB degradation). Use `torch.autocast` for computation only.

### Search Command

```bash
# Training experiments (need >=24GB, fast GPU)
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>=0.95' -o 'dph'

# Eval-only (cheaper OK)
vastai search offers 'gpu_ram>=15 num_gpus=1 dph<=0.15 cuda_vers>=12.0 reliability>=0.95' -o 'dph'
```

### vastai CLI Path (Windows)

```bash
VASTAI="C:/Users/swapp/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/Scripts/vastai.exe"
```

## Experiment Documentation

See `dashboard/EXPERIMENT_TEMPLATE.md` for the format. One file per experiment, plan + results together.

## Submitting PRs to Upstream (openai/parameter-golf)

- Upstream: `upstream` → `https://github.com/openai/parameter-golf.git`
- Fork: `origin` → `https://github.com/swapp1990/parameter-golf.git`
- PRs must be clean: only files under `records/`
- Create branch off `upstream/main`, cherry-pick only submission files, push to origin
- `gh pr create --repo openai/parameter-golf --head swapp1990:<branch> --base main`
