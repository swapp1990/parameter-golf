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

## Experiment Documentation

See `dashboard/EXPERIMENT_TEMPLATE.md` for the format. One file per experiment, plan + results together.

## Submitting PRs to Upstream (openai/parameter-golf)

- Upstream: `upstream` → `https://github.com/openai/parameter-golf.git`
- Fork: `origin` → `https://github.com/swapp1990/parameter-golf.git`
- PRs must be clean: only files under `records/`
- Create branch off `upstream/main`, cherry-pick only submission files, push to origin
- `gh pr create --repo openai/parameter-golf --head swapp1990:<branch> --base main`
