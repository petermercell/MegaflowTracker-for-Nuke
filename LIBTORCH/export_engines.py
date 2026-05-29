#!/usr/bin/env python
"""
export_engines.py — export the five GPU-ready TorchScript engines that the C++
megaflow_solve plugin loads (e_vit, e_cnn, e_head, e_proj, e_refine). Each is a
self-contained `.pt` with a clean one-tensor-per-name IO contract that the C++
TorchEngine (map<name,Tensor> -> positional) drives directly. The plugin loads
them from a directory at runtime — they are not embedded (e_vit alone is 4.6 GB).

Engines:
  * e_vit    — extract_feature, emits ONE stacked token tensor [N,B,S,P,C]
               (no 24-way list — keeps the ViT->head handoff a single tensor).
  * e_cnn    — backbone, returns (cnn0, cnn1).
  * e_head   — stacked tokens + images + cnn0 + cnn1 -> feat. Unbinds the tokens
               list internally so the engine has 4 plain tensor inputs.
  * e_proj   — refine_proj (1x1 conv) traced at a tile (size-generic).
  * e_refine — RAFTUpdateBlock, traced on CUDA at a tile (size-generic) so the
               saved graph loads onto the GPU rather than the CPU.
Every engine is reloaded and numerically re-checked after saving; e_head also
gets a half-window (S//2) check, which pinned a real bug at trace time.

Precision: bf16 autocast by default — matches the model's native compute path on
Ampere+ and the bf16 reference inference. Weights stay fp32 on disk; the engines
run under autocast at use time. fp16 is selectable (--dtype fp16) but diverges
from the model.

Plate aspect: e_vit and e_head bake resize_h × resize_w from --plate_w/--plate_h,
so they are NOT spatial-generic. The 1920x1080 default (16:9) is what 854x480,
3840x2160 etc. all resize to — any 16:9 plate works from one export. Different
aspect ratios (4:3, 21:9, square) need a re-export with matching --plate_w/
--plate_h. e_cnn/e_proj/e_refine are size-generic and unaffected.

Environment: run inside the megaflow311 conda env (PyTorch 2.7.1+cu128, to match
Nuke 17's bundled libtorch ABI), on a CUDA GPU with ~12 GB free for the e_vit
trace. Sitting next to megaflow_cache.py so `from megaflow...` resolves.

    python export_engines.py
    python export_engines.py --plate_w 1920 --plate_h 1080
    python export_engines.py --plate_w 1920 --plate_h 1440  # 4:3 re-export
"""

import argparse
import os
import gc

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

from megaflow.model import MegaFlow
from megaflow.utils.basic import InputPadderMF as InputPadder


# ---------------------------------------------------------------------------
# geometry (mirror forward_track_sliding)
# ---------------------------------------------------------------------------
def model_input_geometry(plate_h, plate_w, patch_size=14):
    padder = InputPadder((1, 1, 3, plate_h, plate_w))
    padded = padder.pad(torch.zeros(1, 3, plate_h, plate_w))
    H_pad, W_pad = padded.shape[-2:]
    resize_w = 518 if patch_size == 14 else 592
    resize_h = round(H_pad * (resize_w / W_pad) / patch_size) * patch_size
    return int(H_pad), int(W_pad), int(resize_h), int(resize_w)


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_AUTOCAST_DTYPE = torch.bfloat16   # set in main() from --dtype

def autocast_ctx():
    return torch.autocast("cuda", dtype=_AUTOCAST_DTYPE)


def flatten(o, out=None):
    out = [] if out is None else out
    if torch.is_tensor(o):
        out.append(o)
    elif isinstance(o, (list, tuple)):
        [flatten(x, out) for x in o]
    return out


def maxdiff(a, b):
    ta, tb = flatten(a), flatten(b)
    if len(ta) != len(tb):
        return float("nan")
    w = 0.0
    for x, y in zip(ta, tb):
        if x.shape != y.shape:
            return float("nan")
        w = max(w, (x.float() - y.float()).abs().max().item())
    return w


def export(name, wrap, example, out_dir, dev):
    """Trace under the configured autocast, save, reload, re-check. Returns
    (path, mb, diff, out_arity)."""
    path = os.path.join(out_dir, f"{name}.pt")
    with torch.no_grad(), autocast_ctx():
        eager = wrap(*example)
        traced = torch.jit.trace(wrap, example, strict=False, check_trace=False)
    traced.save(path)
    mb = os.path.getsize(path) / 1024**2
    # reload exactly as C++ will (onto the GPU) and re-run
    rel = torch.jit.load(path, map_location=dev)
    with torch.no_grad(), autocast_ctx():
        rout = rel(*example)
    d = maxdiff(eager, rout)
    arity = len(flatten(eager))
    del eager, traced, rel
    free()
    print(f"  [{name:<14}] saved {mb:8.1f} MB | reload diff {d:.2e} | outputs {arity}")
    return path, mb, d, arity


# ---------------------------------------------------------------------------
# engine wrappers (each registers ONLY what it needs -> no weight bloat)
# ---------------------------------------------------------------------------
class VitExport(nn.Module):
    """extract_feature -> ONE stacked token tensor [N, B, S, P, C]."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, img):
        return torch.stack(self.m.extract_feature(img), 0)


class HeadExport(nn.Module):
    """stacked tokens + images + cnn0 + cnn1 -> feat. Unbinds tokens & rebuilds
    the cnn list internally so the engine has 4 plain tensor inputs."""
    def __init__(self, m):
        super().__init__()
        self.flow_head = m.flow_head
        self.psi = int(m.patch_start_idx)
    def forward(self, tokens_stacked, images, cnn0, cnn1):
        tokens_list = list(torch.unbind(tokens_stacked, 0))
        # frames_chunk_size=10000 (>= any S) forces flow_head's single-pass branch
        # (flow_head.py:28). The default 8 bakes a 2-chunk loop at S=16 that breaks
        # when the sliding path feeds S//2=8 frames -> reshape [0,-1,2048]. No-chunk
        # is frame-count generic and numerically identical (just one pass).
        return self.flow_head(tokens_list, images=images,
                              patch_start_idx=self.psi,
                              cnn_features=[cnn0, cnn1],
                              frames_chunk_size=10000)


class BackboneExport(nn.Module):
    def __init__(self, m): super().__init__(); self.backbone = m.backbone
    def forward(self, frames):
        return self.backbone(frames)


class ProjExport(nn.Module):
    def __init__(self, m): super().__init__(); self.refine_proj = m.refine_proj
    def forward(self, refine_feat):
        return self.refine_proj(refine_feat)


class RefineExport(nn.Module):
    def __init__(self, m, bs): super().__init__(); self.refine = m.refine; self.bs = int(bs)
    def forward(self, net, inp, corr, flow):
        net_o, up_masks, residual, _ = self.refine(net, inp, corr, flow, batch_size=self.bs)
        return net_o, up_masks, residual


def first_conv_in_channels(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.in_channels)
    raise RuntimeError("no Conv2d found in refine_proj")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plate_w", type=int, default=1920)
    p.add_argument("--plate_h", type=int, default=1080)
    p.add_argument("--out_dir", default="engines")
    p.add_argument("--refine_tile", type=int, default=96,
                   help="spatial tile for the GPU refine/proj traces (size-generic)")
    p.add_argument("--backbone_full", action="store_true",
                   help="trace backbone at full plate res (17GB peak) instead of a tile")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                   help="autocast precision to trace under. bf16 matches the model's "
                        "native compute path on Ampere+ (what megaflow_cache uses); "
                        "fp16 has more mantissa bits but differs from the model.")
    args = p.parse_args()

    global _AUTOCAST_DTYPE
    _AUTOCAST_DTYPE = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("WARNING: no CUDA — refine will be a CPU trace again. Run on the A5000.")
    if args.dtype == "bf16" and dev == "cuda" and not torch.cuda.is_bf16_supported():
        print("WARNING: bf16 not supported on this GPU; the model would fall back to "
              "fp16 too — consider --dtype fp16.")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Tracing under autocast {args.dtype}")

    print(f"Loading megaflow-track on {dev} ...")
    model = MegaFlow.from_pretrained("megaflow-track", device=dev).eval()
    S = int(model.seq_len); ps = int(model.patch_size); rf = int(model.refine_factor)
    fc = int(getattr(model, "feature_channels", 128))
    H_pad, W_pad, rh, rw = model_input_geometry(args.plate_h, args.plate_w, ps)
    print(f"  S={S} patch={ps} refine_factor={rf} feat_ch={fc} | "
          f"pad={W_pad}x{H_pad} resize={rw}x{rh}")

    def rnd(*s): return torch.randn(*s, device=dev)

    info = {}
    cnn_shapes = None
    tok_shape = None

    # ---- 1) e_vit : extract_feature with heads nulled, stacked output ----
    print("\n[1/5] e_vit (ViT-L, stacked tokens) ...")
    heads = {k: getattr(model, k) for k in
             ("backbone", "flow_head", "refine", "refine_proj") if hasattr(model, k)}
    for k in heads:
        setattr(model, k, nn.Identity())
    try:
        img = rnd(1, S, 3, rh, rw)
        with torch.no_grad(), autocast_ctx():
            tok_shape = tuple(torch.stack(model.extract_feature(img), 0).shape)
        info["e_vit"] = export("e_vit", VitExport(model), (img,), args.out_dir, dev)
        del img
    finally:
        for k, v in heads.items():
            setattr(model, k, v)
    free()
    print(f"        stacked token shape = {tok_shape}  (N,B,S,P,C)")

    # ---- 2) e_cnn : backbone (capture real cnn shapes for the head synth) ----
    print("\n[2/5] e_cnn (ResNetFPN) ...")
    if args.backbone_full:
        frames = rnd(S, 3, H_pad, W_pad)
    else:
        # size-generic -> trace on a small tile, but we still need REAL full-res
        # cnn output shapes for the head input. Get them from a quick full-res
        # forward, then trace on the tile.
        with torch.no_grad(), autocast_ctx():
            full = model.backbone(rnd(S, 3, H_pad, W_pad))
        cnn_shapes = [tuple(t.shape) for t in flatten(full)]
        del full; free()
        frames = rnd(2, 3, 512, 512)
    info["e_cnn"] = export("e_cnn", BackboneExport(model), (frames,), args.out_dir, dev)
    if cnn_shapes is None:  # full-res path: recompute real shapes for head
        with torch.no_grad(), autocast_ctx():
            full = model.backbone(rnd(S, 3, H_pad, W_pad))
        cnn_shapes = [tuple(t.shape) for t in flatten(full)]
        del full
    del frames; free()
    print(f"        cnn level shapes = {cnn_shapes}")
    assert len(cnn_shapes) == 2, f"expected 2 FPN levels, got {len(cnn_shapes)}"

    # ---- 3) e_head : stacked tokens + images + cnn0 + cnn1 -> feat ----
    print("\n[3/5] e_head (FlowFeature) ...")
    toks = rnd(*tok_shape)
    img = rnd(1, S, 3, rh, rw)
    cnn0 = rnd(*cnn_shapes[0]); cnn1 = rnd(*cnn_shapes[1])
    info["e_head"] = export("e_head", HeadExport(model), (toks, img, cnn0, cnn1),
                            args.out_dir, dev)
    # half-window check: the sliding path feeds flow_head S//2 frames. Trace at S,
    # but confirm the reloaded engine also runs at S//2 (this is what crashed
    # before the frames_chunk_size fix).
    try:
        half = S // 2
        rel = torch.jit.load(os.path.join(args.out_dir, "e_head.pt"), map_location=dev)
        th = rnd(tok_shape[0], tok_shape[1], half, tok_shape[3], tok_shape[4])
        ih = rnd(1, half, 3, rh, rw)
        c0 = rnd(half, cnn_shapes[0][1], cnn_shapes[0][2], cnn_shapes[0][3])
        c1 = rnd(half, cnn_shapes[1][1], cnn_shapes[1][2], cnn_shapes[1][3])
        with torch.no_grad(), autocast_ctx():
            oh = rel(th, ih, c0, c1)
        print(f"        half-window check (S//2={half}): OK  feat {tuple(oh.shape)}")
        del rel, th, ih, c0, c1, oh
    except Exception as e:
        print(f"        half-window check FAILED: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
    del toks, img, cnn0, cnn1; free()

    # ---- 4) e_proj : refine_proj (1x1 conv) ----
    print("\n[4/5] e_proj (refine_proj) ...")
    proj_in = first_conv_in_channels(model.refine_proj)
    t = args.refine_tile
    info["e_proj"] = export("e_proj", ProjExport(model),
                            (rnd(S, proj_in, t, t),), args.out_dir, dev)
    free()
    print(f"        refine_proj in_channels = {proj_in}")

    # ---- 5) e_refine : RAFTUpdateBlock on CUDA at a tile ----
    print("\n[5/5] e_refine (RAFTUpdateBlock, CUDA tile) ...")
    corr_ch = (2 * 4 + 1) ** 2  # local_radius=4 -> 81
    net = rnd(S, fc, t, t); inp = rnd(S, fc, t, t)
    corr = rnd(S, corr_ch, t, t); flow = rnd(S, 2, t, t)
    info["e_refine"] = export("e_refine", RefineExport(model, 1),
                              (net, inp, corr, flow), args.out_dir, dev)
    del net, inp, corr, flow; free()

    # ---- summary + C++ wiring ----
    total = sum(v[1] for v in info.values())
    print("\n" + "=" * 72)
    print(f"5 engines in ./{args.out_dir}/  —  total {total/1024:.2f} GB on disk")
    for k, v in info.items():
        flag = "  <-- big (ViT weights)" if v[1] > 500 else ""
        print(f"  {k:<10} {v[1]:8.1f} MB   reload-diff {v[2]:.1e}{flag}")
    print("=" * 72)

    print("""
TorchEngine wiring (paste into buildPipeline; names map IO by position):

  e_vit    = mk("e_vit",    {"img"},
                            {"tokens"});
  e_cnn    = mk("e_cnn",    {"frames"},
                            {"cnn0","cnn1"});
  e_head   = mk("e_head",   {"tokens","img","cnn0","cnn1"},
                            {"feat"});
  e_proj   = mk("e_proj",   {"refine_feat"},
                            {"proj"});
  e_refine = mk("e_refine", {"net","inp","corr","flow"},
                            {"net","up_masks","residual_flows"});

C++ handoff is single-tensor throughout:
  tokens = e_vit({img})["tokens"];                       // [N,B,S,P,C]
  cnn    = e_cnn({frames});                               // cnn0, cnn1
  feat   = e_head({tokens,img,cnn["cnn0"],cnn["cnn1"]})["feat"];
  ... global_correlation_softmax (C++) -> flows_init ...
  proj   = e_proj({refine_feat})["proj"]; chunk -> net,inp
  for it: corr=local_correlation_with_flow(C++); e_refine({net,inp,corr,flow})
""")


if __name__ == "__main__":
    main()
