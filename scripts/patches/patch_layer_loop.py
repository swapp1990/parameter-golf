"""Add layer looping to train_gpt.py.

New env var: NUM_UNIQUE_LAYERS (default = NUM_LAYERS, no looping)
When NUM_UNIQUE_LAYERS < NUM_LAYERS:
  - Only NUM_UNIQUE_LAYERS Block instances are created
  - Forward pass uses NUM_LAYERS effective steps, mapping to unique layers cyclically
  - Skip weights remain unique per effective position
"""

with open("/workspace/parameter-golf/train_gpt.py", "r") as f:
    code = f.read()

# 1. Add NUM_UNIQUE_LAYERS env var to hyperparams (after NUM_LAYERS line)
old_env = '    num_layers = int(os.environ.get("NUM_LAYERS", 9))'
new_env = '    num_layers = int(os.environ.get("NUM_LAYERS", 9))\n    num_unique_layers = int(os.environ.get("NUM_UNIQUE_LAYERS", 0))  # 0 = same as num_layers'
assert old_env in code, "Could not find NUM_LAYERS env var"
code = code.replace(old_env, new_env)

# 2. Pass num_unique_layers to GPT constructor — find the GPT(...) call
old_gpt_call = """    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )"""

new_gpt_call = """    if args.num_unique_layers == 0:
        args.num_unique_layers = args.num_layers
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_unique_layers=args.num_unique_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )"""

assert old_gpt_call in code, "Could not find GPT() constructor call"
code = code.replace(old_gpt_call, new_gpt_call)

# 3. Modify GPT.__init__ to accept num_unique_layers and create fewer blocks
old_init_sig = """    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):"""

new_init_sig = """    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        num_unique_layers: int = 0,
    ):"""

assert old_init_sig in code, "Could not find GPT.__init__ signature"
code = code.replace(old_init_sig, new_init_sig)

# 4. Replace block creation to use num_unique_layers and add layer_map
old_blocks = """        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )"""

new_blocks = """        # Layer looping: create fewer unique blocks, reuse them cyclically
        self.num_unique_layers = num_unique_layers if num_unique_layers > 0 else num_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(self.num_unique_layers)
            ]
        )
        # Map effective layer index -> unique block index
        self.layer_map = [i % self.num_unique_layers for i in range(num_layers)]"""

assert old_blocks in code, "Could not find blocks creation"
code = code.replace(old_blocks, new_blocks)

# 5. Modify forward pass to use layer_map
old_forward_encoder = """        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)"""

new_forward_encoder = """        for i in range(self.num_encoder_layers):
            x = self.blocks[self.layer_map[i]](x, x0)
            skips.append(x)"""

assert old_forward_encoder in code, "Could not find encoder loop"
code = code.replace(old_forward_encoder, new_forward_encoder)

old_forward_decoder = """            x = self.blocks[self.num_encoder_layers + i](x, x0)"""

new_forward_decoder = """            x = self.blocks[self.layer_map[self.num_encoder_layers + i]](x, x0)"""

assert old_forward_decoder in code, "Could not find decoder loop"
code = code.replace(old_forward_decoder, new_forward_decoder)

# 6. Add logging for layer looping config
old_log = '    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")'
new_log = '    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")\n    log0(f"num_unique_layers:{args.num_unique_layers} num_effective_layers:{args.num_layers} layer_map:{base_model.layer_map}")'

assert old_log in code, "Could not find attention_mode log line"
code = code.replace(old_log, new_log)

with open("/workspace/parameter-golf/train_gpt.py", "w") as f:
    f.write(code)

print("Layer looping patch applied successfully")
print(f"File length: {len(code.splitlines())} lines")
