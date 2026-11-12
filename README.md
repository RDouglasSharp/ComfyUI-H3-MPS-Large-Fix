# ComfyUI-H3-MPS-Large-Fix

A ComfyUI custom node for Apple Silicon (MPS). It fixes **flat-black or garbage MiniMax H3 output at long sequence lengths**: anything above about 37,400 tokens, for example t2v at 832×480 with more than ~300 frames, or ref2va with a full reference at 1088×608.

It adds no graph nodes. It patches one function when ComfyUI starts.

> **The real fix is upgrading PyTorch.** The cause is a PyTorch 2.12 MPS bug: unary ops such as `silu` on a column-slice view of a bf16 tensor larger than 2^31 bytes return wrong values. PyTorch 2.13.0 and 2.14.1 fix it. With torch 2.14.1 and no node, the 362-frame H3 run below gives the same result as with the node, checked at step 1. Use this node only if you need to stay on torch 2.12.

## The bug

H3's MLP calls `comfy.ops.linear_input_act(fc2, fc1(x), "swiglu")`, a fused SwiGLU with an INT8 down-projection. On MPS, this call returns wrong values for **every row** once its input tensor is larger than 2^31 bytes. It raises no error and produces no NaN.

For H3 that input is S × 28,672 bf16, which crosses the limit at S > 37,449 tokens. The error grows through the 50 DiT blocks until the decoded video is black.

- H3 started hitting it with ComfyUI #16130 (first released in v0.35.0, found by bisect), which routed its MLP through this path on MPS. Seen with torch 2.12.1 on ComfyUI v0.37.2 and v0.44.2, with ComfyUI-AppleSilicon-FP8 1.4.0 and 1.5.0.
- The INT8 matmul and a plain `F.linear` at these sizes are correct. The wrong values come from the SwiGLU step: `silu(gate)`, where `gate` is a column view into the large input. See the box above.

## The fix

On MPS, when the input to `linear_input_act` is larger than 2^31−1 bytes, the node:

1. splits the input into row chunks of at most 2^29 elements;
2. clones each chunk into its own buffer;
3. runs the original function on each chunk;
4. concatenates the results.

Every operation inside the function works row by row, so the result is unchanged. Smaller inputs and other devices take the original path.

## Results

Tested on an M5 Max (128 GB), PyTorch 2.12.1, ComfyUI 0.37.2, MiniMax H3 INT8, 20 steps:

| Clip | Tokens (S) | Without the fix | With the fix |
|---|---|---|---|
| t2v 832×480, 362 f | 43,076 | diverges by block 10 (step-1 probe) | correct (`h3_check` pass) |
| ref2va 832×480, 124 f (1088×608 reference) | ~42k | flat black | correct, 2 of 2 runs |
| ref2va 1088×608, 124 f, full reference | 51,412 | flat black | correct, 2 of 2 runs |
| t2v 1088×608, 277 f | 54,036 | not run | correct at step 1 (probe) |

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/RDouglasSharp/ComfyUI-H3-MPS-Large-Fix
```

Restart ComfyUI. The log should show:

```
[h3_mps_large_fix] active: linear_input_act row-chunked above 2^31-1 bytes on MPS ...
```

The first oversized call also logs `chunking linear_input_act: ...`.

To turn it off, set `H3_MPS_LARGE_FIX=0` in the environment, or set `"enabled": false` in `settings.json`. The environment variable takes precedence.

## Scope

It applies to any model that calls `linear_input_act`, but only on MPS and only above the size limit. It has only been tested with MiniMax H3. On machines without MPS it logs `inactive` and does nothing.

On torch 2.13 or later, the node isn't needed.

Upstream report: [ComfyUI #15799 comment](https://github.com/Comfy-Org/ComfyUI/issues/15799#issuecomment-6805388911)

## License

MIT. Use at your own risk.
