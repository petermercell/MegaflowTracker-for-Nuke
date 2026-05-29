## Build from source

Tested on Rocky Linux 9 with an RTX A5000 (24 GB), CUDA 12.8, and Nuke 17.0v1.

### 1. Set up the Python environment

The exporter runs in a dedicated conda environment with PyTorch 2.7.1 + CUDA
12.8 — this **must** match the libtorch bundled with Nuke 17 so the traced
engines load inside the plugin.

```bash
conda create -n megaflow311 python=3.11
conda activate megaflow311
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install MegaFlow and verify the model runs

Clone the upstream MegaFlow repo and install it editable. The first demo run
auto-downloads the `megaflow-track` weights from Hugging Face.

```bash
git clone https://github.com/cvg/megaflow.git
cd megaflow
pip install -e . --ignore-requires-python
python demo_track.py --input assets/apple.mp4 --grid_size 8
```

If the demo writes a tracked output without errors, the Python side is good.

### 3. Export the engines

From the same environment, run the production exporter. It writes five `.pt`
files into `./engines/` (~4.6 GB total — `e_vit` alone is most of that).

```bash
python export_engines.py
```

Defaults are 1920×1080 (16:9). **If your plates are a different aspect ratio**
(4:3, 21:9, square), re-export with matching `--plate_w/--plate_h` — `e_vit`
and `e_head` bake `resize_h × resize_w` at trace time. See the docstring at
the top of `export_engines.py` for the full precision and aspect notes.

### 4. Build the C++ plugin

```bash
cd /path/to/MegaFlowSolve   # this repo
rm -rf build
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces `build/MegaFlowSolve.so` (the Nuke plugin) and
`build/mf_solve` (the standalone CLI / validation harness).

### 5. Install into Nuke

Copy the plugin into your Nuke plugin path. The exact destination depends on
your platform — for Linux:

```bash
cp build/MegaFlowSolve.so ~/.nuke/CPP/17.0/MegaFlowSolve.so
```

Point the node's `engines` knob at the directory from step 3, or set
`MEGAFLOW_ENGINES` in your environment.

---

### VRAM monitoring (optional)

While solving, you can watch VRAM use with `nvidia-smi`:

```bash
# live, one line per second:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# log to a file in the background:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1 > /tmp/vram.log &

# stop the background logger:
pkill -f "nvidia-smi"
```
