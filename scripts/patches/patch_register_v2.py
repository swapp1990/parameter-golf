"""Patch train_gpt.py with two register token approaches:
  REGISTER_MODE=0: disabled (baseline)
  REGISTER_MODE=1: append register at end, inject back via learned gate
  REGISTER_MODE=2: bidirectional register at position 0
"""

with open("/workspace/parameter-golf/train_gpt.py", "r") as f:
    code = f.read()

# 1. Modify CausalSelfAttention.forward to accept optional attn_mask
old_attn_forward = """    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)"""

new_attn_forward = """    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if attn_mask is not None:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=False,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)"""

assert old_attn_forward in code, "Could not find old attn forward"
code = code.replace(old_attn_forward, new_attn_forward)

# 2. Modify Block.forward to pass attn_mask
old_block_forward = """    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x"""

new_block_forward = """    def forward(self, x: Tensor, x0: Tensor, attn_mask: Tensor = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x"""

assert old_block_forward in code, "Could not find old block forward"
code = code.replace(old_block_forward, new_block_forward)

# 3. Replace init register section
old_init = """        # Register token: learnable global context vector
        self.use_register_token = bool(int(os.environ.get("USE_REGISTER_TOKEN", "0")))
        if self.use_register_token:
            self.register_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        # Context injection: mean-pool global context added after each layer
        self.use_context_injection = bool(int(os.environ.get("USE_CONTEXT_INJECTION", "0")))
        if self.use_context_injection:
            self.ctx_scale = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])"""

new_init = """        # Register token approaches (env var controlled)
        # REGISTER_MODE=0: disabled (baseline)
        # REGISTER_MODE=1: append register at end, inject back via gate
        # REGISTER_MODE=2: bidirectional register at position 0
        self.register_mode = int(os.environ.get("REGISTER_MODE", "0"))
        if self.register_mode in (1, 2):
            self.register_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        if self.register_mode == 1:
            self.reg_gate = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])"""

assert old_init in code, "Could not find old init register"
code = code.replace(old_init, new_init)

# 4. Replace forward body
old_forward = '        x = self.tok_emb(input_ids)\n        x = F.rms_norm(x, (x.size(-1),))\n\n        # Prepend register token if enabled\n        if self.use_register_token:\n            reg = self.register_token.expand(x.size(0), 1, x.size(-1)).to(dtype=x.dtype)\n            x = torch.cat([reg, x], dim=1)\n\n        x0 = x\n        skips: list[Tensor] = []\n\n        # First half stores skips; second half reuses them in reverse order.\n        for i in range(self.num_encoder_layers):\n            x = self.blocks[i](x, x0)\n            # Context injection in encoder too\n            if self.use_context_injection:\n                global_ctx = x.mean(dim=1, keepdim=True)\n                x = x + self.ctx_scale[i] * global_ctx\n            skips.append(x)\n        for i in range(self.num_decoder_layers):\n            if skips:\n                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()\n            x = self.blocks[self.num_encoder_layers + i](x, x0)\n            # Context injection: add mean-pooled global context\n            if self.use_context_injection:\n                layer_idx = self.num_encoder_layers + i\n                global_ctx = x.mean(dim=1, keepdim=True)\n                x = x + self.ctx_scale[layer_idx] * global_ctx\n\n        # Strip register token before output projection\n        if self.use_register_token:\n            x = x[:, 1:, :]'

new_forward = '        x = self.tok_emb(input_ids)\n        x = F.rms_norm(x, (x.size(-1),))\n\n        attn_mask = None\n        if self.register_mode == 1:\n            # Append register token at the end - causal masking lets it see everything\n            reg = self.register_token.expand(x.size(0), 1, x.size(-1)).to(dtype=x.dtype)\n            x = torch.cat([x, reg], dim=1)  # [B, seq+1, D]\n        elif self.register_mode == 2:\n            # Prepend register token at position 0 with bidirectional attention\n            reg = self.register_token.expand(x.size(0), 1, x.size(-1)).to(dtype=x.dtype)\n            x = torch.cat([reg, x], dim=1)  # [B, 1+seq, D]\n            # Build mask: register (row 0) sees all, all see register (col 0), rest causal\n            seqlen = x.size(1)\n            causal = torch.tril(torch.ones(seqlen, seqlen, device=x.device, dtype=x.dtype))\n            causal[0, :] = 1.0  # register row: bidirectional\n            # col 0 already 1 from tril\n            attn_mask = torch.where(causal == 1.0,\n                                    torch.tensor(0.0, device=x.device, dtype=x.dtype),\n                                    torch.tensor(float(\"-inf\"), device=x.device, dtype=x.dtype))\n            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]\n\n        x0 = x\n        skips: list[Tensor] = []\n\n        for i in range(self.num_encoder_layers):\n            x = self.blocks[i](x, x0, attn_mask=attn_mask)\n            if self.register_mode == 1:\n                reg_hidden = x[:, -1:, :]  # register at last pos\n                x = x + self.reg_gate[i] * reg_hidden\n            skips.append(x)\n        for i in range(self.num_decoder_layers):\n            if skips:\n                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()\n            layer_idx = self.num_encoder_layers + i\n            x = self.blocks[layer_idx](x, x0, attn_mask=attn_mask)\n            if self.register_mode == 1:\n                reg_hidden = x[:, -1:, :]\n                x = x + self.reg_gate[layer_idx] * reg_hidden\n\n        # Strip register token before output\n        if self.register_mode == 1:\n            x = x[:, :-1, :]  # remove appended register\n        elif self.register_mode == 2:\n            x = x[:, 1:, :]   # remove prepended register'

assert old_forward in code, "Could not find old forward body"
code = code.replace(old_forward, new_forward)

with open("/workspace/parameter-golf/train_gpt.py", "w") as f:
    f.write(code)

print("Patch applied successfully")
print(f"File length: {len(code.splitlines())} lines")
