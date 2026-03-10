import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCH_DIR = ROOT / "architecture"
SETTING_DIR = ROOT / "setting"


def _get_opt_from_argv(argv: list[str], long_name: str, short_name: str, default: str) -> str:
    """Parse argv for --long / -short before full parse."""
    for i, arg in enumerate(argv):
        if arg in (f"--{long_name}", short_name) and i + 1 < len(argv):
            return argv[i + 1]
    return default


def load_json_config(base_dir: Path, name: str) -> dict:
    """Load base_dir/<name>.json. Return {} if missing or invalid."""
    path = base_dir / f"{name}.json"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _cmd_atrium3d(args: argparse.Namespace) -> int:
    # `atrium3d/atrium3d.py` currently contains the `Atrium3D` class + `visualize()`.
    # Keep this entrypoint lightweight: only use the visualization feature.
    from atrium3d.atrium3d import Atrium3D

    a = Atrium3D(
        benchmark=args.benchmark,
        dir=args.dir,
        type=args.type,
        size=args.size,
        layers=args.layers,
        spacing_xy=args.spacing_xy,
        spacing_z=args.spacing_z,
        routing_steps_per_move=args.routing_steps_per_move,
        routing_pause_frames=args.routing_pause_frames,
    )
    save_path = str(ROOT / args.save) if args.save else None
    a.visualize(save_path=save_path)
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    """
    Run the minimum compilable process: read benchmark -> scheduling -> placing -> save results JSON.
    (routing/simulate/animate is not implemented in this repository, I can add them later if needed)
    """
    from atrium3d.atrium3d import Atrium3D

    a = Atrium3D(
        benchmark=args.benchmark,
        dir=args.dir,
        type=args.type,
        size=args.size,
        layers=args.layers,
        spacing_xy=args.spacing_xy,
        spacing_z=args.spacing_z,
        routing_steps_per_move=args.routing_steps_per_move,
        routing_pause_frames=args.routing_pause_frames,
        scheduling_strategy=args.scheduling_strategy,
    )
    # Pass readout heuristic weight via results_code to avoid widening the constructor too much.
    # (Atrium3D will pick it up in solve()).
    a.results_code["readout_weight"] = float(args.readout_weight)
    rc = a.solve(simulation=False, animation=False, do_routing=False, initial_zone=args.initial_zone)
    # Print a compact summary
    print(f"[INFO] compile done: n_qubits={rc['n_qubits']}, n_stages={rc['n_stages']}")
    return 0


def _cmd_stage_frames(args: argparse.Namespace) -> int:
    """
    Run scheduling + placing + stage placement, and export a PNG for each micro-stage.
    """
    from atrium3d.atrium3d import Atrium3D

    a = Atrium3D(
        benchmark=args.benchmark,
        dir=args.dir,
        type=args.type,
        size=args.size,
        layers=args.layers,
        spacing_xy=args.spacing_xy,
        spacing_z=args.spacing_z,
        routing_steps_per_move=args.routing_steps_per_move,
        routing_pause_frames=args.routing_pause_frames,
        scheduling_strategy=args.scheduling_strategy,
    )
    a.results_code["readout_weight"] = float(args.readout_weight)
    a.solve(simulation=False, animation=False, do_routing=False, initial_zone=args.initial_zone)

    return 0


def _cmd_animate(args: argparse.Namespace) -> int:

    from atrium3d.atrium3d import Atrium3D

    a = Atrium3D(
        benchmark=args.benchmark,
        dir=args.dir,
        type=args.type,
        size=args.size,
        layers=args.layers,
        spacing_xy=args.spacing_xy,
        spacing_z=args.spacing_z,
        routing_steps_per_move=args.routing_steps_per_move,
        routing_pause_frames=args.routing_pause_frames,
        scheduling_strategy=args.scheduling_strategy,
    )
    a.results_code["readout_weight"] = float(args.readout_weight)
    # In solve(), routing() and generate_animation() will be called automatically.
    a.solve(
        simulation=False,
        animation=True,
        do_routing=True,
        initial_zone=args.initial_zone,
    )
    return 0

# One table for all CLI options (defaults here are fallbacks; JSON overrides before parse).
# Keys must match architecture/setting JSON keys. Add "choices" when needed.
OPTIONS = {
    "benchmark": (str, "qft_n10", "benchmark name (no extension)"),
    "dir": (str, "default", "benchmark subdirectory in benchmark/"),
    "type": (str, "qasm", "benchmark file format", {"choices": ["qasm", "json"]}),
    "size": (int, 7, "XY grid size"),
    "layers": (int, 6, "number of Z layers"),
    "spacing_xy": (float, 5.0, "XY spacing (µm) between trap sites"),
    "spacing_z": (float, 25.0, "Z spacing (µm) between layers"),
    "scheduling_strategy": (str, "asap", "scheduling strategy"),
    "initial_zone": (str, "storage", "initial atom zone", {"choices": ["storage", "all"]}),
    "readout_weight": (float, 0.0, "readout proximity heuristic weight"),
    "routing_steps_per_move": (int, 15, "interpolation steps per move"),
    "routing_pause_frames": (int, 5, "pause frames at gate"),
    "save": (str, "architecture.png", "output image path (atrium3d)"),
    "out_dir": (str, "stage_frames", "output dir for frames"),
    "prefix": (str, "stage_", "filename prefix for frames"),
    "every": (int, 1, "export every N micro-stages"),
    "max_frames": (int, None, "max frames to export (None = all)"),
    "dpi": (int, 150, "PNG dpi"),
}


def _add_options(parser: argparse.ArgumentParser, names: list[str], config: dict | None) -> None:
    """Add arguments for `names` from OPTIONS; apply defaults from config when present."""
    for name in names:
        spec = OPTIONS[name]
        type_, default, help_str = spec[0], spec[1], spec[2]
        kwargs = {"type": type_, "default": (config or {}).get(name, default), "help": help_str}
        if len(spec) == 4:
            kwargs.update(spec[3])
        parser.add_argument(f"--{name}", **kwargs)


def _apply_config_to_parser(parser_or_sub: argparse.ArgumentParser, config: dict) -> None:
    """Set default for each option that exists in config and in this (sub)parser."""
    if not config:
        return
    for action in parser_or_sub._actions:
        if action.dest != "help" and action.dest in config:
            action.default = config[action.dest]


def build_parser(config: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3D-NAA project entry script (adapted for current directory structure)",
    )
    parser.add_argument("--setting", "-s", type=str, default="default", help="setting/<name>.json (includes which architecture to use)")
    sub = parser.add_subparsers(dest="cmd", required=False)

    atrium_opts = ["size", "layers", "spacing_xy", "spacing_z", "benchmark", "dir", "type", "save"]
    run_opts = ["benchmark", "dir", "type", "size", "layers", "spacing_xy", "spacing_z", "scheduling_strategy", "initial_zone", "readout_weight", "routing_steps_per_move", "routing_pause_frames"]
    frames_extra = ["out_dir", "prefix", "every", "max_frames", "dpi"]

    p_atrium = sub.add_parser("atrium3d", help="generate 3D architecture diagram")
    _add_options(p_atrium, atrium_opts, config)
    p_atrium.set_defaults(func=_cmd_atrium3d)

    p_compile = sub.add_parser("compile", help="run scheduling+placing (save results JSON)")
    _add_options(p_compile, run_opts, config)
    p_compile.set_defaults(func=_cmd_compile)

    p_frames = sub.add_parser("frames", help="export PNG per micro-stage")
    _add_options(p_frames, run_opts + frames_extra, config)
    p_frames.set_defaults(func=_cmd_stage_frames)

    p_anim = sub.add_parser("animate", help="full pipeline + 3D routing MP4")
    _add_options(p_anim, run_opts, config)
    p_anim.set_defaults(func=_cmd_animate)

    if config:
        _apply_config_to_parser(p_atrium, config)
        _apply_config_to_parser(p_compile, config)
        _apply_config_to_parser(p_frames, config)
        _apply_config_to_parser(p_anim, config)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    setting_name = _get_opt_from_argv(argv, "setting", "-s", "default")
    setting_config = load_json_config(SETTING_DIR, setting_name)
    arch_name = setting_config.get("architecture", "default")
    arch_config = load_json_config(ARCH_DIR, arch_name)
    # Merge: architecture (structure) + setting (benchmark & run options); setting overrides
    config = {**arch_config, **setting_config}
    parser = build_parser(config)
    args = parser.parse_args(argv)
    if getattr(args, "cmd", None) is None:
        args = parser.parse_args(["atrium3d"] + argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
