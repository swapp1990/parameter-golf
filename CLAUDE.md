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

## RunPod Pod Management

- When switching GPUs, use shared/network volumes so data persists across pods.
- Always terminate old pods before creating new ones.
- Never guess Docker image tags — use known working images or look them up first.
- When creating pods, verify the image exists before deploying.
