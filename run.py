from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


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
    )
    save_path = str(ROOT / args.save) if args.save else None
    a.visualize(save_path=save_path)
    return 0


def _cmd_view(_: argparse.Namespace) -> int:
    # Runs the reference visualization in `view.py`.
    import view

    view.visualize_na_atrium_architecture()
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    # `test.py` is a self-contained demo script (has top-level code).
    # Run it in a subprocess to avoid import side-effects.
    p = subprocess.run([sys.executable, str(ROOT / "test.py")], cwd=str(ROOT))
    return int(p.returncode)


def _cmd_compile(args: argparse.Namespace) -> int:
    """
    运行最小可用的编译流程：读取 benchmark -> scheduling -> placing -> 保存 results JSON。
    （routing/simulate/animate 当前仓库未实现，后续需要我可以继续补齐）
    """
    from atrium3d.atrium3d import Atrium3D

    a = Atrium3D(
        benchmark=args.benchmark,
        dir=args.dir,
        type=args.type,
        size=args.size,
        layers=args.layers,
        scheduling_strategy=args.scheduling_strategy,
    )
    # Pass readout heuristic weight via results_code to avoid widening the constructor too much.
    # (Atrium3D will pick it up in solve()).
    a.results_code["readout_weight"] = float(args.readout_weight)
    rc = a.solve(simulation=False, animation=False, do_routing=False, initial_zone=args.initial_zone)
    # Print a compact summary
    print(f"[INFO] compile done: n_qubits={rc['n_qubits']}, n_stages={rc['n_stages']}")
    return 0


def _cmd_mapping(args: argparse.Namespace) -> int:
    """
    运行 scheduling+placing，并把 initial_mapping 画出来（默认保存到项目根目录）。
    """
    from atrium3d.atrium3d import Atrium3D

    a = Atrium3D(
        benchmark=args.benchmark,
        dir=args.dir,
        type=args.type,
        size=args.size,
        layers=args.layers,
        scheduling_strategy=args.scheduling_strategy,
    )
    a.results_code["readout_weight"] = float(args.readout_weight)
    a.solve(simulation=False, animation=False, do_routing=False, initial_zone=args.initial_zone)
    save_path = str(ROOT / args.save) if args.save else None
    a.visualize_initial_mapping(save_path=save_path, show=bool(args.show))
    print(f"[INFO] mapping plot saved: {save_path}" if save_path else "[INFO] mapping plot done")
    return 0


def _cmd_stage_frames(args: argparse.Namespace) -> int:
    """
    运行 scheduling+placing+stage placement，并导出每个 micro-stage 的 PNG。
    """
    from atrium3d.atrium3d import Atrium3D

    a = Atrium3D(
        benchmark=args.benchmark,
        dir=args.dir,
        type=args.type,
        size=args.size,
        layers=args.layers,
        scheduling_strategy=args.scheduling_strategy,
    )
    a.results_code["readout_weight"] = float(args.readout_weight)
    a.solve(simulation=False, animation=False, do_routing=False, initial_zone=args.initial_zone)

    out_dir = ROOT / args.out_dir
    generated = a.save_all_micro_stage_images(
        out_dir=out_dir,
        prefix=args.prefix,
        every=args.every,
        max_frames=args.max_frames,
        dpi=args.dpi,
    )
    print(f"[INFO] generated {len(generated)} frames in {out_dir}")
    return 0


def _cmd_animate(args: argparse.Namespace) -> int:
    """
    运行完整编译 + 路由 + 3D 动画生成（GIF）。
    """
    from atrium3d.atrium3d import Atrium3D

    a = Atrium3D(
        benchmark=args.benchmark,
        dir=args.dir,
        type=args.type,
        size=args.size,
        layers=args.layers,
        scheduling_strategy=args.scheduling_strategy,
    )
    a.results_code["readout_weight"] = float(args.readout_weight)
    # 在 solve() 里会自动调用 routing() + generate_animation()
    a.solve(
        simulation=False,
        animation=True,
        do_routing=True,
        initial_zone=args.initial_zone,
    )
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3D-NAA project entry script (adapted for current directory structure)",
    )
    # Allow `python run.py` with no subcommand; we'll default to `atrium3d`.
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_atrium = sub.add_parser("atrium3d", help="generate Atrium 3D architecture diagram (default architecture)")
    p_atrium.add_argument("--size", type=int, default=10, help="XY grid size")
    p_atrium.add_argument("--layers", type=int, default=4, help="number of layers in Z direction")
    p_atrium.add_argument("--benchmark", type=str, default="qft_n10", help="benchmark name (without extension)")
    p_atrium.add_argument("--dir", type=str, default="default", help="benchmark subdirectory (in benchmark/)")
    p_atrium.add_argument("--type", type=str, default="qasm", choices=["qasm", "json"], help="default benchmark file format")
    p_atrium.add_argument("--save", type=str, default="atrium3d.png", help="output image file name (in project root)")
    p_atrium.set_defaults(func=_cmd_atrium3d)

    p_view = sub.add_parser("view", help="run view.py's visualization (show in a window)")
    p_view.set_defaults(func=_cmd_view)

    p_demo = sub.add_parser("demo", help="run test.py's demo scene (show in a window)")
    p_demo.set_defaults(func=_cmd_demo)

    p_compile = sub.add_parser("compile", help="run scheduling+placing (save results JSON)")
    p_compile.add_argument("--benchmark", type=str, default="qft_n10", help="benchmark name (without extension)")
    p_compile.add_argument("--dir", type=str, default="default", help="benchmark subdirectory (in benchmark/)")
    p_compile.add_argument("--type", type=str, default="qasm", choices=["qasm", "json"], help="default benchmark file format")
    p_compile.add_argument("--size", type=int, default=7, help="default trap grid size (for default architecture)")
    p_compile.add_argument("--layers", type=int, default=6, help="number of layers (for default architecture)")
    p_compile.add_argument("--scheduling_strategy", type=str, default="asap", help="scheduling strategy (currently only asap is supported)")
    p_compile.add_argument("--initial_zone", type=str, default="storage", choices=["storage", "all"], help="initial atom location (default storage)")
    p_compile.add_argument("--readout_weight", type=float, default=0.0, help="readout proximity heuristic weight (0 to disable)")
    p_compile.set_defaults(func=_cmd_compile)

    p_map = sub.add_parser("mapping", help="visualize initial_mapping (save PNG)")
    p_map.add_argument("--benchmark", type=str, default="qft_n10", help="benchmark name (without extension)")
    p_map.add_argument("--dir", type=str, default="default", help="benchmark subdirectory (in benchmark/)")
    p_map.add_argument("--type", type=str, default="qasm", choices=["qasm", "json"], help="default benchmark file format")
    p_map.add_argument("--size", type=int, default=7, help="default trap grid size (for default architecture)")
    p_map.add_argument("--layers", type=int, default=6, help="number of layers (for default architecture)")
    p_map.add_argument("--scheduling_strategy", type=str, default="asap", help="scheduling strategy (currently only asap is supported)")
    p_map.add_argument("--initial_zone", type=str, default="storage", choices=["storage", "all"], help="initial atom location (default storage)")
    p_map.add_argument("--readout_weight", type=float, default=0.0, help="readout proximity heuristic weight (0 to disable)")
    p_map.add_argument("--save", type=str, default="initial_mapping.png", help="output image file name (in project root)")
    p_map.add_argument("--show", action="store_true", help="whether to show in a window (default only save)")
    p_map.set_defaults(func=_cmd_mapping)

    p_frames = sub.add_parser("frames", help="export PNG for every micro-stage")
    p_frames.add_argument("--benchmark", type=str, default="qft_n10", help="benchmark name (without extension)")
    p_frames.add_argument("--dir", type=str, default="default", help="benchmark subdirectory (in benchmark/)")
    p_frames.add_argument("--type", type=str, default="qasm", choices=["qasm", "json"], help="default benchmark file format")
    p_frames.add_argument("--size", type=int, default=7, help="default trap grid size (for default architecture)")
    p_frames.add_argument("--layers", type=int, default=6, help="number of layers (for default architecture)")
    p_frames.add_argument("--scheduling_strategy", type=str, default="asap", help="scheduling strategy (currently only asap is supported)")
    p_frames.add_argument("--initial_zone", type=str, default="storage", choices=["storage", "all"], help="initial atom location (default storage)")
    p_frames.add_argument("--readout_weight", type=float, default=0.0, help="readout proximity heuristic weight (0 to disable)")
    p_frames.add_argument("--out_dir", type=str, default="stage_frames", help="output directory (relative to project root)")
    p_frames.add_argument("--prefix", type=str, default="stage_", help="file name prefix")
    p_frames.add_argument("--every", type=int, default=1, help="export every N micro-stages")
    p_frames.add_argument("--max_frames", type=int, default=None, help="only export first N frames (optional)")
    p_frames.add_argument("--dpi", type=int, default=250, help="PNG dpi")
    p_frames.set_defaults(func=_cmd_stage_frames)

    p_anim = sub.add_parser("animate", help="run full pipeline and generate 3D routing GIF")
    p_anim.add_argument("--benchmark", type=str, default="qft_n10", help="benchmark name (without extension)")
    p_anim.add_argument("--dir", type=str, default="default", help="benchmark subdirectory (in benchmark/)")
    p_anim.add_argument("--type", type=str, default="qasm", choices=["qasm", "json"], help="default benchmark file format")
    p_anim.add_argument("--size", type=int, default=7, help="default trap grid size (for default architecture)")
    p_anim.add_argument("--layers", type=int, default=6, help="number of layers (for default architecture)")
    p_anim.add_argument("--scheduling_strategy", type=str, default="asap", help="scheduling strategy (currently only asap is supported)")
    p_anim.add_argument("--initial_zone", type=str, default="storage", choices=["storage", "all"], help="initial atom location (default storage)")
    p_anim.add_argument("--readout_weight", type=float, default=0.0, help="readout proximity heuristic weight (0 to disable)")
    p_anim.set_defaults(func=_cmd_animate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "cmd", None) is None:
        # Default behavior: generate the Atrium 3D architecture image.
        args = parser.parse_args(["atrium3d"])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
