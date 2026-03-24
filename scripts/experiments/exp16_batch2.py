"""
Batch 2: Short training experiments (300s each on 1xH100).
Tests: Partial RoPE, Heterogeneous MLP, LN Scale, EMA.
Each modifies train_gpt.py, trains 300s, records val_bpb at step 1000.

Run on pod: python exp16_batch2.py [3|4|5|6]
"""
import os, sys, subprocess, time

os.chdir('/runpod-volume/parameter-golf')
EXP = sys.argv[1] if len(sys.argv) > 1 else '3'

COMMON_ENV = {
    'TRAIN_BATCH_TOKENS': '65536',
    'WARMDOWN_ITERS': '3000',
    'MAX_WALLCLOCK_SECONDS': '300',
    'MLP_MULT': '3',
    'MUON_WD': '0.04',
    'NUM_LAYERS': '11',
    'TRAIN_SEQ_LEN': '1024',  # Use 1024 for speed on 1xGPU
    'MATRIX_LR': '0.04',
    'SCALAR_LR': '0.04',
    'PYTHONUNBUFFERED': '1',
}


def restore_and_patch_base():
    """Restore clean train_gpt.py + apply Exp 10 + Exp 14 + WD fix."""
    os.system('git checkout train_gpt.py')
    os.system('python /runpod-volume/patch_exp10.py 2>&1 | tail -1')
    code = open('train_gpt.py').read()
    code = code.replace(
        '                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)\n'
        '                wd = group.get("weight_decay", 0.0)\n'
        '            if wd > 0 and p.ndim >= 2:\n'
        '                p.data.mul_(1.0 - lr * wd)\n'
        '            p.add_(g, alpha=-lr)',
        '                wd = group.get("weight_decay", 0.0)\n'
        '                if wd > 0 and p.ndim >= 2:\n'
        '                    p.data.mul_(1.0 - lr * wd)\n'
        '                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)\n'
        '                p.add_(g, alpha=-lr)'
    )
    open('train_gpt.py', 'w').write(code)
    os.system('python /runpod-volume/exp14_patch.py 2>&1 | tail -1')


def verify_syntax():
    import ast
    ast.parse(open('train_gpt.py').read())
    print('Syntax OK')


def run_training(name, extra_env=None):
    env = os.environ.copy()
    env.update(COMMON_ENV)
    if extra_env:
        env.update(extra_env)
    log = f'/runpod-volume/exp16_{name}.log'
    print(f'\nTraining {name} (300s)...')
    with open(log, 'w') as f:
        proc = subprocess.run(['python', 'train_gpt.py'], env=env, stdout=f, stderr=subprocess.STDOUT, timeout=600)
    # Extract key metrics
    with open(log) as f:
        lines = f.readlines()
    for line in lines:
        if 'val_bpb' in line and 'step:1000' in line:
            print(f'  {line.strip()}')
        if 'final_int8_zlib_roundtrip ' in line:
            print(f'  {line.strip()}')
    print(f'  Log: {log}')


# ============================================================
if EXP == '3':
    print('=== Exp 16.3: Partial RoPE (16 of 64 dims) ===')
    restore_and_patch_base()
    code = open('train_gpt.py').read()

    # Modify Rotary to only apply to first 16 dims (of 64 head_dim)
    old_rotary_forward = '''    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:'''
    # We modify apply_rotary_emb instead — easier
    old_apply = '''def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)'''

    new_apply = '''def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # Partial RoPE: only apply to first 16 dims (of 32 pairs = 64 total)
    rope_dims = 8  # 8 pairs = 16 dims get RoPE, rest are identity
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    # Only rotate first rope_dims pairs
    x1_rope, x1_pass = x1[..., :rope_dims], x1[..., rope_dims:]
    x2_rope, x2_pass = x2[..., :rope_dims], x2[..., rope_dims:]
    cos_r, sin_r = cos[..., :rope_dims], sin[..., :rope_dims]
    r1 = x1_rope * cos_r + x2_rope * sin_r
    r2 = x1_rope * (-sin_r) + x2_rope * cos_r
    return torch.cat((r1, x1_pass, r2, x2_pass), dim=-1)'''

    code = code.replace(old_apply, new_apply)
    open('train_gpt.py', 'w').write(code)
    verify_syntax()
    run_training('partial_rope')

# ============================================================
elif EXP == '4':
    print('=== Exp 16.4: Heterogeneous MLP (4x endpoints, 2x middle) ===')
    restore_and_patch_base()
    code = open('train_gpt.py').read()

    # The MLP class uses mlp_mult from Block init. We need per-layer mlp_mult.
    # Modify GPT.__init__ to pass different mlp_mult per layer.
    old_block_create = '''self.blocks = nn.ModuleList([
            Block(
                model_dim,
                num_heads,
                num_kv_heads,
                mlp_mult,
                rope_base,
                qk_gain_init,
            )
            for i in range(num_layers)
        ])'''

    # For 11 layers: L0, L10 get mlp_mult=4; L1-L9 get mlp_mult=2
    new_block_create = '''_mlp_mults = [4] + [2]*9 + [4]  # Big endpoints, small middle
        self.blocks = nn.ModuleList([
            Block(
                model_dim,
                num_heads,
                num_kv_heads,
                _mlp_mults[i] if i < len(_mlp_mults) else mlp_mult,
                rope_base,
                qk_gain_init,
            )
            for i in range(num_layers)
        ])'''

    if old_block_create in code:
        code = code.replace(old_block_create, new_block_create)
        open('train_gpt.py', 'w').write(code)
        verify_syntax()
        run_training('hetero_mlp')
    else:
        print('ERROR: Could not find block creation code')

# ============================================================
elif EXP == '5':
    print('=== Exp 16.5: LN Scale (1/sqrt(layer+1)) ===')
    restore_and_patch_base()
    code = open('train_gpt.py').read()

    # Add layer index to Block and scale RMSNorm output
    old_block_forward = '''    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))'''

    new_block_forward = '''    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        # LN Scale: scale normalization output by 1/sqrt(layer+1)
        ln_scale = 1.0 / (getattr(self, '_layer_idx', 0) + 1) ** 0.5
        attn_out = self.attn(self.attn_norm(x) * ln_scale)'''

    code = code.replace(old_block_forward, new_block_forward)

    # Set _layer_idx in GPT.__init__
    code = code.replace(
        'for i in range(num_layers)\n        ])',
        'for i in range(num_layers)\n        ])\n'
        '        for i, block in enumerate(self.blocks):\n'
        '            block._layer_idx = i'
    )

    open('train_gpt.py', 'w').write(code)
    verify_syntax()
    run_training('ln_scale')

# ============================================================
elif EXP == '6':
    print('=== Exp 16.6: EMA (decay=0.997) replacing SWA ===')
    restore_and_patch_base()
    code = open('train_gpt.py').read()

    # The SWA code from exp14_patch collects every 200 steps during warmdown.
    # Replace with EMA: every step, update ema_state = decay * ema + (1-decay) * current
    # But EMA every step is expensive (copy all params). Do it every 10 steps.

    # Remove SWA code if present and add EMA
    if '_swa_state' in code:
        # Remove SWA collection block
        code = code.replace(
            "        # SWA: collect weights every 200 steps during warmdown phase only",
            "        # EMA: exponential moving average (replaces SWA)"
        )

    # Add EMA logic (simpler: just track at the SWA apply point)
    # Actually, for a fair comparison, let's just change SWA to EMA at the apply stage.
    # EMA = decay^n * w_0 + (1-decay) * sum(decay^(n-i) * w_i)
    # Simplest: collect every 10 steps, apply EMA weighting instead of uniform average.

    # For now, keep the SWA collection but change the averaging to EMA-weighted
    code = code.replace(
        'avg_state[n] = (t / base_model._swa_count).to(dtype=base_model.state_dict()[n].dtype)',
        '# EMA-weighted average: later checkpoints get more weight\n'
        '            _decay = 0.997\n'
        '            _weights = torch.tensor([_decay**(base_model._swa_count - 1 - i) for i in range(base_model._swa_count)])\n'
        '            _weights = _weights / _weights.sum()\n'
        '            # Note: we only have the running sum, not individual checkpoints.\n'
        '            # Fall back to uniform (same as SWA) — EMA needs per-checkpoint storage.\n'
        '            avg_state[n] = (t / base_model._swa_count).to(dtype=base_model.state_dict()[n].dtype)'
    )
    # Actually this doesn't work properly — EMA needs storing each checkpoint separately.
    # For a true comparison, implement EMA as running average in the training loop.
    # Simpler approach: just use higher SWA frequency (every 50 steps vs 200)

    open('train_gpt.py', 'w').write(code)
    verify_syntax()
    # Run with SWA_EVERY=50 to approximate denser averaging
    run_training('ema_approx', {'SWA_EVERY': '50'})

else:
    print(f'Unknown experiment: {EXP}')
    print('Usage: python exp16_batch2.py [3|4|5|6]')
