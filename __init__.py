"""h3_mps_large_fix: row-chunk comfy.ops.linear_input_act on MPS when its input exceeds 2**31-1 bytes.

Since ComfyUI #16130 (first released in v0.35.0), MiniMax H3's MLP runs
linear_input_act(fc2, fc1(x), "swiglu") through comfy_kitchen's eager int8_linear on MPS. Once
fc1's output (S x 28,672 bf16) passes 2**31 bytes (S > 37,449 tokens), that call returns wrong
values for every row, with no error and no NaN. The error compounds over the 50 blocks into
flat-black output. Running the call in row chunks, each cloned into its own buffer, gives the
correct result. Every op inside (input activation, per-row INT8 activation quantization, matmul,
per-row scale and the optional residual epilogue) is row-wise, so chunking does not change the math.

Applies to any caller of linear_input_act, but only on MPS and only above the size limit; every
other call takes the original path untouched. Tested with MiniMax H3 only.
Kill switch: H3_MPS_LARGE_FIX=0, or "enabled": false in settings.json (the environment wins).
No graph nodes are registered.
"""
import json
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
TAG = "[h3_mps_large_fix]"

_LIMIT_BYTES = (1 << 31) - 1      # above this, the MPS path misbehaves
_CHUNK_ELEMS = 1 << 29            # per-chunk input stays far below it


def _enabled():
    env = os.environ.get("H3_MPS_LARGE_FIX")
    if env is not None:
        return env.strip().lower() not in ("", "0", "false", "no", "off"), f"H3_MPS_LARGE_FIX={env}"
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")) as f:
            on = bool(json.load(f).get("enabled", True))
        return on, f"settings.json enabled={on}"
    except Exception:
        return True, "default"


def _install():
    import inspect
    import torch
    import comfy.ops as ops

    orig = ops.linear_input_act
    if getattr(orig, "_h3_mps_large_fix", False):
        return "already installed"
    sig = inspect.signature(orig)
    params = list(sig.parameters)
    x_name = "x" if "x" in params else params[1]
    state = {"logged": False}

    def _rows_view(t, lead, rows):
        if not isinstance(t, torch.Tensor) or t.dim() < 2:
            return None          # scalars / per-channel vectors broadcast unchanged
        try:
            return t.expand(*lead, t.shape[-1]).reshape(rows, t.shape[-1])
        except RuntimeError:
            return None

    def linear_input_act(*args, **kwargs):
        ba = sig.bind(*args, **kwargs)
        x = ba.arguments.get(x_name)
        if not (isinstance(x, torch.Tensor) and x.device.type == "mps"
                and x.numel() * x.element_size() > _LIMIT_BYTES):
            return orig(*args, **kwargs)
        width = x.shape[-1]
        lead = tuple(x.shape[:-1])
        rows = x.numel() // max(width, 1)
        flat = x.reshape(rows, width)
        per = max(1, _CHUNK_ELEMS // max(width, 1))
        row_args = {}
        for name in ("residual", "residual_scale"):
            if ba.arguments.get(name) is not None:
                rv = _rows_view(ba.arguments[name], lead, rows)
                if rv is not None:
                    row_args[name] = rv
        if not state["logged"]:
            state["logged"] = True
            print(f"{TAG} chunking linear_input_act: input {tuple(x.shape)} {x.dtype} "
                  f"({x.numel() * x.element_size() / 2**30:.2f} GiB) in {-(-rows // per)} row chunks",
                  flush=True)
        outs = []
        for i in range(0, rows, per):
            cb = sig.bind(*ba.args, **ba.kwargs)
            cb.arguments[x_name] = flat[i:i + per].clone()
            for name, rv in row_args.items():
                cb.arguments[name] = rv[i:i + per].clone()
            outs.append(orig(*cb.args, **cb.kwargs))
        out = torch.cat(outs, dim=0)
        return out.reshape(*lead, out.shape[-1])

    linear_input_act._h3_mps_large_fix = True
    linear_input_act.__wrapped__ = orig
    ops.linear_input_act = linear_input_act
    return "installed"


_on, _why = _enabled()
if _on:
    try:
        import torch
        if not torch.backends.mps.is_available():
            print(f"{TAG} inactive (MPS not available)", flush=True)
        else:
            print(f"{TAG} active: linear_input_act row-chunked above 2^31-1 bytes on MPS ({_why}; "
                  f"{_install()})", flush=True)
    except Exception as e:
        print(f"{TAG} inactive (install failed: {e!r}); stock behaviour", flush=True)
else:
    print(f"{TAG} inactive ({_why})", flush=True)
