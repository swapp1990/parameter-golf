"""
Layer X-Ray: Interactive visualization of what each transformer layer learns.

Usage:
    streamlit run dashboard/layer_xray.py

Requires checkpoints in dashboard/checkpoints/ directory:
    model_step_10.pt, model_step_50.pt, ..., final_model.pt
And tokenizer at data/tokenizers/fineweb_1024_bpe.model
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm
import streamlit as st

# ---------------------------------------------------------------------------
# Model definition (standalone, no imports from train_gpt.py)
# ---------------------------------------------------------------------------

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x, return_attn=False):
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

        if return_attn:
            # Manual attention to capture weights
            # Expand KV heads for GQA
            heads_per_kv = self.num_heads // self.num_kv_heads
            if heads_per_kv > 1:
                k = k.repeat_interleave(heads_per_kv, dim=1)
                v = v.repeat_interleave(heads_per_kv, dim=1)
            scale = 1.0 / (self.head_dim ** 0.5)
            attn = (q @ k.transpose(-2, -1)) * scale
            # Causal mask
            mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask[None, None, :, :], float('-inf'))
            attn_weights = F.softmax(attn, dim=-1)
            y = attn_weights @ v
            y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
            return self.proj(y), attn_weights
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                                enable_gqa=(self.num_kv_heads != self.num_heads))
            y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
            return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0, return_attn=False):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if return_attn:
            attn_out, attn_weights = self.attn(self.attn_norm(x), return_attn=True)
        else:
            attn_out = self.attn(self.attn_norm(x))
            attn_weights = None
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        if return_attn:
            return x, attn_weights
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=1024, num_layers=9, model_dim=512, num_heads=8,
                 num_kv_heads=4, mlp_mult=2, tie_embeddings=True,
                 tied_embed_init_std=0.02, logit_softcap=30.0,
                 rope_base=10000.0, qk_gain_init=1.5):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

    def get_logits(self, x):
        """Convert hidden states to logits."""
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward_full(self, input_ids, skip_layers=None, return_attn=False, return_hidden=False):
        """
        Full forward pass with options:
        - skip_layers: set of layer indices to skip (for knockout)
        - return_attn: return attention weights per layer
        - return_hidden: return hidden states after each layer
        """
        skip_layers = skip_layers or set()
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        skips = []
        attn_maps = {}
        hidden_states = [x.detach().clone()]

        # Encoder
        for i in range(self.num_encoder_layers):
            if i in skip_layers:
                skips.append(torch.zeros_like(x))
            else:
                if return_attn:
                    x, aw = self.blocks[i](x, x0, return_attn=True)
                    attn_maps[i] = aw.detach()
                else:
                    x = self.blocks[i](x, x0)
                skips.append(x)
            if return_hidden:
                hidden_states.append(x.detach().clone())

        # Decoder
        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            if layer_idx in skip_layers:
                pass  # skip this layer
            else:
                if return_attn:
                    x, aw = self.blocks[layer_idx](x, x0, return_attn=True)
                    attn_maps[layer_idx] = aw.detach()
                else:
                    x = self.blocks[layer_idx](x, x0)
            if return_hidden:
                hidden_states.append(x.detach().clone())

        logits = self.get_logits(x)
        result = {"logits": logits}
        if return_attn:
            result["attn"] = attn_maps
        if return_hidden:
            result["hidden"] = hidden_states
        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@st.cache_resource
def load_tokenizer(path):
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


@st.cache_resource
def load_model(checkpoint_path):
    model = GPT()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def tokenize(sp, text, vocab_size=1024):
    ids = sp.Encode(text)
    ids = [min(t, vocab_size - 1) for t in ids]
    return ids


def decode_token(sp, token_id):
    try:
        return sp.IdToPiece(token_id)
    except:
        return f"[{token_id}]"


def get_top_predictions(logits, sp, k=10):
    """Get top-k predictions from logits at last position."""
    probs = F.softmax(logits[0, -1, :].float(), dim=-1)
    topk = torch.topk(probs, k)
    results = []
    for i in range(k):
        tid = topk.indices[i].item()
        prob = topk.values[i].item()
        token_str = decode_token(sp, tid)
        results.append((token_str, prob, tid))
    return results


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Layer X-Ray", layout="wide")
    st.title("Layer X-Ray")
    st.caption("Interactive visualization of what each transformer layer learns")

    # --- Sidebar: Configuration ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(project_root, "data", "tokenizers", "fineweb_1024_bpe.model")
    checkpoint_dir = os.path.join(project_root, "dashboard", "checkpoints")

    if not os.path.exists(tokenizer_path):
        st.error(f"Tokenizer not found at: {tokenizer_path}")
        st.stop()

    if not os.path.exists(checkpoint_dir):
        st.error(f"Checkpoint directory not found: {checkpoint_dir}")
        st.info("Place model checkpoints (model_step_*.pt or final_model.pt) in dashboard/checkpoints/")
        st.stop()

    # Find available checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    if not ckpt_files:
        st.error("No checkpoint files found in dashboard/checkpoints/")
        st.stop()

    sp = load_tokenizer(tokenizer_path)

    # Parse checkpoint names for display
    def ckpt_label(path):
        name = os.path.basename(path)
        if "step_" in name:
            step = name.split("step_")[1].replace(".pt", "")
            return f"Step {step}"
        if "final" in name:
            return "Final"
        return name.replace(".pt", "")

    ckpt_labels = {ckpt_label(f): f for f in ckpt_files}

    st.sidebar.header("Settings")
    selected_ckpt = st.sidebar.selectbox("Checkpoint", list(ckpt_labels.keys()),
                                          index=len(ckpt_labels) - 1)

    input_text = st.sidebar.text_area("Input text", value="The cat sat on the", height=100)

    view_mode = st.sidebar.radio("View", ["Layer Knockout", "Training Timeline", "Attention Map"])

    # --- Tokenize input ---
    token_ids = tokenize(sp, input_text)
    if len(token_ids) == 0:
        st.warning("Enter some text to analyze")
        st.stop()

    token_strs = [decode_token(sp, t) for t in token_ids]
    st.sidebar.markdown(f"**Tokens ({len(token_ids)}):** `{'` `'.join(token_strs)}`")

    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    # =====================================================================
    # VIEW 1: Layer Knockout
    # =====================================================================
    if view_mode == "Layer Knockout":
        st.header("Layer Knockout")
        st.markdown("What happens to predictions when each layer is removed?")

        model = load_model(ckpt_labels[selected_ckpt])

        with torch.no_grad():
            # Baseline (all layers)
            base_result = model.forward_full(input_tensor)
            base_preds = get_top_predictions(base_result["logits"], sp, k=5)

            st.subheader("All layers (baseline)")
            cols = st.columns(5)
            for i, (tok, prob, _) in enumerate(base_preds):
                cols[i].metric(f"#{i+1}", tok.replace("▁", " "), f"{prob:.1%}")

            st.divider()
            st.subheader("With each layer removed")

            # Knockout each layer
            knockout_data = []
            for layer_idx in range(model.num_layers):
                result = model.forward_full(input_tensor, skip_layers={layer_idx})
                preds = get_top_predictions(result["logits"], sp, k=5)

                # Compute KL divergence from baseline
                base_probs = F.softmax(base_result["logits"][0, -1, :].float(), dim=-1)
                ko_probs = F.softmax(result["logits"][0, -1, :].float(), dim=-1)
                kl = F.kl_div(ko_probs.log(), base_probs, reduction="sum").item()

                knockout_data.append({
                    "layer": layer_idx,
                    "preds": preds,
                    "kl": kl,
                    "top1_match": preds[0][0] == base_preds[0][0],
                    "top1_token": preds[0][0],
                    "top1_prob": preds[0][1],
                })

            # Show as table
            for item in knockout_data:
                layer_type = "Encoder" if item["layer"] < model.num_encoder_layers else "Decoder"
                if item["layer"] == model.num_encoder_layers - 1 or item["layer"] == model.num_encoder_layers:
                    layer_type = "Bottleneck"

                match_icon = "" if item["top1_match"] else " (CHANGED)"
                impact = item["kl"]
                impact_bar = "█" * min(int(impact * 10), 30)

                col1, col2, col3, col4 = st.columns([1, 2, 3, 2])
                col1.markdown(f"**L{item['layer']}** ({layer_type})")
                col2.markdown(f"`{item['top1_token'].replace('▁', ' ')}` ({item['top1_prob']:.1%}){match_icon}")
                col3.markdown(f"`{impact_bar}` KL={impact:.4f}")

                other_preds = " ".join([f"`{t.replace('▁', ' ')}`({p:.0%})" for t, p, _ in item["preds"][1:3]])
                col4.markdown(other_preds)

    # =====================================================================
    # VIEW 2: Training Timeline
    # =====================================================================
    elif view_mode == "Training Timeline":
        st.header("Training Timeline")
        st.markdown("How predictions evolve as the model trains")

        timeline_data = []
        for label, path in ckpt_labels.items():
            model = load_model(path)
            with torch.no_grad():
                result = model.forward_full(input_tensor)
                preds = get_top_predictions(result["logits"], sp, k=5)

                # Also get loss if we have > 1 token
                loss = None
                if len(token_ids) > 1:
                    logits = result["logits"][0, :-1, :]  # shift
                    targets = input_tensor[0, 1:]  # shift
                    loss = F.cross_entropy(logits.float(), targets).item()

                timeline_data.append({
                    "label": label,
                    "preds": preds,
                    "loss": loss,
                })

        # Show evolution
        for item in timeline_data:
            st.subheader(item["label"])
            loss_str = f" | Loss: {item['loss']:.4f} ({item['loss'] / np.log(2) * 0.3584:.4f} BPB)" if item["loss"] else ""
            cols = st.columns(6)
            for i, (tok, prob, _) in enumerate(item["preds"][:5]):
                cols[i].metric(f"#{i+1}", tok.replace("▁", " "), f"{prob:.1%}")
            if loss_str:
                cols[5].markdown(f"**{loss_str}**")
            st.divider()

        # Training Timeline + Layer Knockout combined
        if len(ckpt_labels) > 1:
            st.subheader("Layer Impact Over Training")
            st.markdown("KL divergence when each layer is knocked out, across checkpoints")

            chart_data = []
            for label, path in ckpt_labels.items():
                model = load_model(path)
                with torch.no_grad():
                    base_result = model.forward_full(input_tensor)
                    base_probs = F.softmax(base_result["logits"][0, -1, :].float(), dim=-1)

                    for layer_idx in range(model.num_layers):
                        result = model.forward_full(input_tensor, skip_layers={layer_idx})
                        ko_probs = F.softmax(result["logits"][0, -1, :].float(), dim=-1)
                        kl = F.kl_div(ko_probs.log(), base_probs, reduction="sum").item()
                        chart_data.append({"Checkpoint": label, "Layer": f"L{layer_idx}", "KL Impact": kl})

            import pandas as pd
            df = pd.DataFrame(chart_data)
            pivot = df.pivot(index="Layer", columns="Checkpoint", values="KL Impact")
            # Reorder columns by step number
            st.dataframe(pivot.style.background_gradient(cmap="YlOrRd", axis=None), use_container_width=True)

    # =====================================================================
    # VIEW 3: Attention Map
    # =====================================================================
    elif view_mode == "Attention Map":
        st.header("Attention Heatmap")
        st.markdown("Where each attention head looks")

        model = load_model(ckpt_labels[selected_ckpt])

        with torch.no_grad():
            result = model.forward_full(input_tensor, return_attn=True)

        layer_idx = st.slider("Layer", 0, model.num_layers - 1, 0)

        if layer_idx in result["attn"]:
            attn = result["attn"][layer_idx][0]  # [num_heads, seq, seq]
            num_heads = attn.shape[0]

            # Show per-head attention to previous token (bigram signal)
            st.subheader("Previous-token attention (bigram signal)")
            prev_attn = []
            for h in range(num_heads):
                # For each position, how much attention goes to position-1?
                diag_below = torch.diagonal(attn[h], offset=-1)
                prev_attn.append(diag_below.mean().item())

            cols = st.columns(num_heads)
            for h in range(num_heads):
                pct = prev_attn[h]
                cols[h].metric(f"Head {h}", f"{pct:.1%}", help="Avg attention to immediately previous token")

            # Show full attention heatmap for selected head
            head_idx = st.slider("Head", 0, num_heads - 1, 0)

            attn_matrix = attn[head_idx].numpy()
            seq_len = len(token_ids)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.5), max(6, seq_len * 0.4)))
            im = ax.imshow(attn_matrix[:seq_len, :seq_len], cmap="Blues", aspect="auto")
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            labels = [t.replace("▁", " ") for t in token_strs]
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("Key (attending to)")
            ax.set_ylabel("Query (attending from)")
            ax.set_title(f"Layer {layer_idx}, Head {head_idx}")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning(f"No attention data for layer {layer_idx} (was it skipped?)")


if __name__ == "__main__":
    main()
