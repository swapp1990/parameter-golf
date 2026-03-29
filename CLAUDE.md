# Parameter Golf Project

## Running Experiments on RunPod

Use the experiment runner script on the pod:

```bash
# Start an experiment (use run_in_background so it doesn't block)
ssh ... "/root/run_experiment.sh <name> ENV_VAR=value ..."

# Example:
ssh ... "/root/run_experiment.sh exp9 REGISTER_MODE=2 WARMDOWN_ITERS=3000 TRAIN_BATCH_TOKENS=65536 MAX_WALLCLOCK_SECONDS=1200"
```

- Logs go to `/root/experiments/<name>.log` — no grep piping, full output visible in background task
- Check progress: `ssh ... "tail -5 /root/experiments/<name>.log"`
- Check status: `ssh ... "cat /root/experiments/<name>.status"`
- Summary at end: automatically prints val_bpb steps and final results

## Background Task Rules

- Only ONE background task per training run
- Do NOT spawn extra "wait" or "poll" loops as background tasks
- Check progress with quick foreground SSH commands when asked
- Always set `TRAIN_BATCH_TOKENS=65536` for single GPU (default 524288 is for 8xH100)

## Step-by-Step Execution (CRITICAL — READ THIS EVERY TIME)

- NEVER chain multiple commands in one SSH call. Run ONE command at a time.
- NEVER run blocking SSH commands that take >15 seconds. Instead:
  1. Run the command on the pod with `nohup ... > /runpod-volume/somefile.log 2>&1 &`
  2. Poll the log file every 15-30 seconds with `tail -5 /runpod-volume/somefile.log`
  3. Show the user the output each time
- Before each step, state what you're doing and roughly how long it should take.
- After each step completes, show the result before moving to the next.
- If something fails, fix it immediately — don't retry blindly.
- Examples of commands that MUST be backgrounded + polled:
  - `pip install` (can take 30-120s)
  - `git clone` (can take 10-60s)
  - `python data/cached_challenge_fineweb.py` (downloads data, 30-120s)
  - `python train_gpt.py` (training, minutes to hours)
  - Any python script that does heavy computation
- Examples of commands that can run directly (fast, <10s):
  - `echo`, `ls`, `cat`, `tail`, `head`, `wc`, `ps`, `nvidia-smi`
  - `git checkout`, `python -c 'import ast; ...'` (syntax checks)
  - `kill`, `mv`, `cp`, `mkdir`

## Checkpoint Management (CRITICAL)

- After EVERY experiment completes, ALWAYS download the checkpoint (final_model.pt) to local machine BEFORE stopping the pod.
- Save to: `D:/MyProjects/Claude/parameter-golf/dashboard/checkpoints/<exp_name>/`
- Also download the training log.
- Never stop a pod without first downloading the checkpoint — it may be on a non-shared volume and lost forever.

## RunPod Pod Management

`runpodctl` is installed at `C:\Users\swapp\AppData\Local\Microsoft\WindowsApps\runpodctl.exe`. API key is pre-configured.

### Creating a Pod

```bash
# List existing pods
runpodctl pod list

# List network volumes
runpodctl network-volume list

# Create 1xH100 SXM pod with network volume (US-MO-1)
runpodctl pod create \
  --name "pod-name" \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --image "runpod/parameter-golf:latest" \
  --network-volume-id "y26gyuaocv" \
  --volume-mount-path "/runpod-volume" \
  --container-disk-in-gb 20 \
  --data-center-ids "US-MO-1" \
  --ports "22/tcp"
```

### Connecting via SSH

```bash
# Get SSH connection info (wait until pod is ready)
runpodctl ssh info <pod-id>

# Connect (uses ssh.runpod.io proxy)
ssh root@<pod-id>-ssh.proxy.runpod.io
```

### Managing Pods

```bash
runpodctl pod stop <pod-id>    # stop (keeps data)
runpodctl pod start <pod-id>   # restart stopped pod
runpodctl pod remove <pod-id>  # terminate permanently
```

### Key Resources

- Network volume: `param-golf-shared` (id: `y26gyuaocv`, 50GB, US-MO-1)
- Volume mount: `/runpod-volume/`
- Docker image: `runpod/parameter-golf:latest`
- GPU IDs: `NVIDIA H100 80GB HBM3` (SXM), `NVIDIA H100 PCIe`
- When switching GPUs, use shared/network volumes so data persists across pods.
- Always terminate old pods before creating new ones.
- Never guess Docker image tags — use known working images or look them up first.
- When creating pods, verify the image exists before deploying.

## Submitting PRs to Upstream (openai/parameter-golf)

- Upstream remote: `upstream` → `https://github.com/openai/parameter-golf.git`
- Fork: `origin` → `https://github.com/swapp1990/parameter-golf.git`
- PRs must be clean: only files under `records/` — no dashboard, scripts, CLAUDE.md, patches, etc.
- To submit cleanly:
  1. `git fetch upstream main`
  2. Create a branch off `upstream/main`
  3. Cherry-pick or copy only the `records/track_*/<submission>/` files
  4. Push branch to `origin`, then `gh pr create --repo openai/parameter-golf --head swapp1990:<branch> --base main`
- Never submit from a branch based on fork's `main` — it carries all local experiment commits.
