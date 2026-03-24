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

## Running Scripts on the Pod (CRITICAL — READ EVERY TIME)

1. **Write and test code LOCALLY first.** Verify it works before touching the pod.
2. **Upload ONE script** to the pod.
3. **Launch in background:** `nohup ... > /runpod-volume/eval.log 2>&1 &`
4. **Give the user a PowerShell-compatible tail command** to monitor themselves:
   ```
   ssh -i C:\Users\swapp\.ssh\id_ed25519 root@<IP> -p <PORT> "tail -f /runpod-volume/eval.log"
   ```
5. **Hands off.** Do NOT poll, sleep, or check the log yourself. Wait for the user to tell you the result or ask you to check.
6. **Never run multiple things simultaneously.** One script, one launch.
7. **When user says done:** download results, stop pod.

### Quick commands (OK to run directly, <10s):
`echo`, `ls`, `cat`, `tail`, `head`, `wc`, `ps`, `nvidia-smi`, `kill`, `mv`, `cp`, `mkdir`, `git checkout`, syntax checks

### Everything else MUST be backgrounded:
`pip install`, `git clone`, `python *.py`, any computation

## Checkpoint Management (CRITICAL)

- After EVERY experiment completes, ALWAYS download the checkpoint (final_model.pt) to local machine BEFORE stopping the pod.
- Save to: `D:/MyProjects/Claude/parameter-golf/dashboard/checkpoints/<exp_name>/`
- Also download the training log.
- Never stop a pod without first downloading the checkpoint — it may be on a non-shared volume and lost forever.

## File Organization (FOLLOW THIS)

Never put scripts or files in the root directory or dashboard root. Use the organized structure:

```
Root (only these .py files):
  train_gpt.py              — original baseline
  train_gpt_submission.py   — submission-ready version
  train_gpt_mlx.py          — MLX variant

scripts/
  patches/      — code patches (patch_exp10.py, exp14_patch.py, build_submission.py, etc.)
  eval/         — evaluation scripts (ttt_eval.py, sliding window, debug, local tests)
  analysis/     — analysis tools (checkpoint_analysis.py, generate_report.py, dashboards)
  experiments/  — experiment-specific scripts (exp11, exp16 batches, quantize)

dashboard/
  plans/                — experiment plans (experiments_plan.md, exp17_plan.md)
  experiment_analyses/  — per-experiment analysis docs (exp10, exp11, exp12-14, exp15, exp16)
  reports/
    README.md           — standard report format
    exp17/              — comprehensive report for exp17 (analysis.md, data, tables)
    exp<N>/             — future experiments follow same structure
  frontend/             — React dashboard (compiled)
```

**Rules:**
- New scripts go in `scripts/<category>/`
- New experiment analyses go in `dashboard/experiment_analyses/`
- Comprehensive reports go in `dashboard/reports/exp<N>/`
- Plans go in `dashboard/plans/`
- NEVER put loose .py or .md files in root or dashboard root
- Use `git rm` for specific files, NEVER `git add -A` on large directories

## RunPod Pod Management

- When switching GPUs, use shared/network volumes so data persists across pods.
- Always terminate old pods before creating new ones.
- Never guess Docker image tags — use known working images or look them up first.
- When creating pods, verify the image exists before deploying.
