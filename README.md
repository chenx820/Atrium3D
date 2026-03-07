# Atrium3D

A neutral-atom 3D compiler: schedule, place, and route quantum circuits on a 3D trap architecture, with support for 3D visualization and animation export.

---

## Quick start

```bash
# Default (no args): generate 3D architecture diagram atrium3d.png
python run.py

# List all subcommands and options
python run.py --help
python run.py compile --help
python run.py mapping --help
python run.py frames --help
python run.py animate --help
```

---

## Subcommands

| Subcommand | Description                                                       |
| ---------- | ----------------------------------------------------------------- |
| `atrium3d` | Generate 3D architecture diagram only (default)                   |
| `compile`  | Schedule + place, write results to JSON (no routing/animation)    |
| `mapping`  | Schedule + place, then draw initial mapping (PNG)                 |
| `frames`   | Schedule + place + stage placement, export a PNG per micro-stage  |
| `animate`  | Full pipeline: schedule + place + route + generate 3D routing MP4 |

---

## Common arguments

These options are shared by `compile`, `mapping`, `frames`, and `animate`.

| Argument                | Default   | Description                                                                 |
| ----------------------- | --------- | --------------------------------------------------------------------------- |
| `--benchmark`           | `qft_n10` | Benchmark name (no extension); file is `benchmark/<dir>/<benchmark>.<type>` |
| `--dir`                 | `default` | Subdirectory under `benchmark/`, e.g. `benchmark/default/`                  |
| `--type`                | `qasm`    | Benchmark file format: `qasm` or `json`                                     |
| `--size`                | see below | XY trap grid size (number of cells)                                         |
| `--layers`              | see below | Number of layers in Z                                                       |
| `--scheduling_strategy` | `asap`    | Scheduling strategy (currently only `asap` is supported)                    |
| `--initial_zone`        | `storage` | Initial atom positions: `storage` or `all`                                  |
| `--readout_weight`      | `0.0`     | Readout proximity heuristic weight; use 0 to disable                        |

**Default `--size` / `--layers`:**

- `atrium3d`: `--size 10`, `--layers 4`
- `compile` / `mapping` / `frames` / `animate`: `--size 7`, `--layers 6`

---

## Subcommand-specific arguments

### `atrium3d`

| Argument | Default        | Description                                  |
| -------- | -------------- | -------------------------------------------- |
| `--save` | `atrium3d.png` | Output image path (relative to project root) |

### `mapping`

| Argument | Default               | Description                                                    |
| -------- | --------------------- | -------------------------------------------------------------- |
| `--save` | `initial_mapping.png` | Path for the initial mapping image                             |
| `--show` | (flag)                | Open a window to display the plot; otherwise only save to file |

### `frames`

| Argument       | Default        | Description                                             |
| -------------- | -------------- | ------------------------------------------------------- |
| `--out_dir`    | `stage_frames` | Output directory (relative to project root)             |
| `--prefix`     | `stage_`       | Filename prefix, e.g. `stage_0001.png`                  |
| `--every`      | `1`            | Export one frame every N micro-stages (1 = every stage) |
| `--max_frames` | (none)         | Export only the first N frames; omit to export all      |
| `--dpi`        | `250`          | PNG resolution (DPI)                                    |

### `animate`

No extra arguments. When the full pipeline finishes, it writes `routing_animation.mp4` in the **project root**.

---

## Examples

```bash
# 1. Only draw 3D architecture, save as my_arch.png
python run.py atrium3d --save my_arch.png --size 8 --layers 5

# 2. Compile a custom benchmark (results to JSON)
python run.py compile --benchmark my_circuit --dir experiments --type qasm --size 7 --layers 6

# 3. Run initial mapping and show the plot in a window
python run.py mapping --benchmark qft_n10 --save qft_mapping.png --show

# 4. Export PNG frames, first 50 only, 300 DPI
python run.py frames --benchmark qft_n10 --out_dir my_frames --max_frames 50 --dpi 300

# 5. Full pipeline and generate routing animation
python run.py animate --benchmark qft_n10 --scheduling_strategy asap --initial_zone storage
```

---

## Config files

Defaults are read from two JSON configs (CLI flags override these):

- **`architecture/<name>.json`** — system structure only: `size`, `layers`, `spacing_xy`, `spacing_z` (µm). Which file is used is chosen by the setting.
- **`setting/<name>.json`** — what to run and how: `architecture` (which `architecture/<name>.json` to load), `benchmark`, `dir`, `type`, `scheduling_strategy`, `initial_zone`, `readout_weight`, `save`, `prefix`, `every`, `max_frames`, `dpi`, `routing_steps_per_move`, `routing_pause_frames`. Use `-s` / `--setting` to choose the setting file (default: `default`).

Example: `python run.py -s default animate` loads `setting/default.json`, which sets `"architecture": "default"` so that `architecture/default.json` is used. Edit the setting file to change the benchmark or which architecture to use.

---

## Paths and layout

- Benchmark path: `benchmark/<dir>/<benchmark>.<type>`  
  Example: `--dir default --benchmark qft_n10 --type qasm` → `benchmark/default/qft_n10.qasm`
- Output paths (`--save`, `--out_dir`) are relative to the project root.

---

## Dependencies

Install Python dependencies from the project’s config (e.g. `requirements.txt` or `pyproject.toml`) before running the commands above.
