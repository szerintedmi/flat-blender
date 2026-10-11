"""`scanmodel` command line. Each stage reads and writes files under data/<project>/ and is skipped when unchanged."""

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from typer._click.globals import get_current_context

from . import testposes
from .cache import Stage, console, fingerprint, run_stage
from .config import POSE_SOURCES, Config, ConfigError, default_root, from_repo, load_config
from .deband import DEBAND_MODES
from .geom import GL_CV
from .io import find_bundles
from .models.derived import IngestReport
from .models.overrides import Overrides
from .points import WALLS_DIR
from .schema import write_schemas
from .stages import eval as eval_stage
from .stages import fuse, ingest, poses, refine, scanmesh, splat, structure
from .stages.stray import import_stray

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
Force = Annotated[bool, typer.Option("--force", "-f", help="Run even if inputs are unchanged.")]
Project = Annotated[
    str | None, typer.Option("--project", help="Project folder under data/ (default: settings.toml project).")
]
Raw = Annotated[
    Path | None,
    typer.Option(help="Folder to ingest instead of data/<project>/raw."),
]
PoseGraph = Annotated[
    bool | None,
    typer.Option(
        "--pose-graph/--no-pose-graph",
        help="Correct ARKit's drift (default: settings.toml pose_graph, else on).",
    ),
]
PosesFrom = Annotated[
    str | None,
    typer.Option(
        "--poses",
        help="Poses for every stage after `poses`: refine (bundle-adjusted by `scanmodel refine`, the default) or"
        " posegraph (`scanmodel poses` alone, as before P7). Default: settings.toml poses, else refine.",
    ),
]
Skip = Annotated[
    list[str] | None,
    typer.Option("--skip", help="Session folder to leave out. Repeatable."),
]


def _cli_project() -> str | None:
    """--project, before or after the command."""
    ctx = get_current_context(silent=True)
    while ctx is not None:
        if project := str(ctx.params.get("project") or "").strip():
            return str(project)
        ctx = ctx.parent
    return None


def _cfg() -> Config:
    try:
        return load_config(project=_cli_project())
    except ConfigError as e:
        raise typer.BadParameter(str(e), param_hint="--project") from e


def _project_line(cfg: Config) -> str:
    """`project   fifth (settings.toml)`: the scan the command works on and which setting chose it."""
    return f"project   {cfg.project} ({cfg.project_source})"


@app.callback()
def main(project: Project = None) -> None:
    """Select a project globally, or pass --project after a command."""


def _overrides(cfg: Config) -> Path:
    return cfg.scene_dir / "overrides.json"


def _drop_frames(cfg: Config) -> dict[str, list[tuple[int, int]]]:
    """overrides.json `drop_frames`: the only part ingest reads (a param, so other edits don't re-ingest)."""
    path = _overrides(cfg)
    return Overrides.model_validate_json(path.read_text()).drop_frames if path.is_file() else {}


def _ingest_stage(cfg: Config, raw: Path | None = None, skip: list[str] | None = None) -> Stage:
    raw = raw.expanduser().resolve() if raw else cfg.raw
    drop = _drop_frames(cfg)
    skipped: list[str] = sorted(set(skip or []))
    bundles = find_bundles(raw, skipped)
    if unknown := set(skipped) - {d.name for d in find_bundles(raw)}:
        console.print(
            f"[yellow]ingest: no session to skip named {', '.join(sorted(unknown))} in {raw}[/yellow]"
        )
    return Stage(
        "ingest",
        inputs=lambda: bundles,
        run=lambda: ingest.run(bundles, cfg.stage_dir("ingest"), drop),
        root=cfg.root,
        params={"raw": cfg.rel(raw), "skip": skipped} | ({"drop_frames": drop} if drop else {}),
        code=["stages/ingest.py", "checks.py", "stillcal.py"],
    )


def _ingested(cfg: Config) -> list[Path]:
    """Raw bundle folders of the sessions the last ingest wrote."""
    reports = sorted(cfg.stage_dir("ingest").glob("*/report.json"))
    return [cfg.from_repo(IngestReport.model_validate_json(p.read_text()).raw_dir) for p in reports]


def _poses_stage(cfg: Config, pose_graph: bool | None = None, stability: bool = False) -> Stage:
    graph = cfg.pose_graph if pose_graph is None else pose_graph
    return Stage(
        "poses",
        inputs=lambda: [cfg.stage_dir("ingest"), _overrides(cfg)],
        run=lambda: poses.run(
            cfg.stage_dir("ingest"), _overrides(cfg), cfg.stage_dir("poses"), graph, stability and graph
        ),
        root=cfg.root,
        params={"pose_graph": graph} | ({"stability": True} if stability and graph else {}),
        code=["stages/poses.py", "points.py", "roomplan.py", "align.py"],
    )


def _refine_stage(
    cfg: Config,
    settle_drop_s: float = refine.SETTLE_DROP_S,
    readout_s: float = 0.0,
    floor_sigma: float = refine.rf.FLOOR_SIGMA,
) -> Stage:
    p = cfg.stage_dir("poses")
    return Stage(
        "refine",
        inputs=lambda: [p / "poses.json", p / "frames.jsonl", p / "stills.jsonl", cfg.stage_dir("ingest")],
        run=lambda: refine.run(
            p, cfg.stage_dir("ingest"), cfg.stage_dir("refine"), settle_drop_s, readout_s, floor_sigma
        ),
        root=cfg.root,
        params={"settle_drop_s": settle_drop_s, "readout_s": readout_s, "floor_sigma": floor_sigma},
        code=["stages/refine.py", "refine.py", "align.py", "points.py", "roomplan.py"],
    )


def _source(cfg: Config, value: str | None) -> str:
    """The pose source to use: --poses, else settings.toml poses, else refine."""
    src = value or cfg.poses
    if src not in POSE_SOURCES:
        raise typer.BadParameter(
            f"pick one of {', '.join(POSE_SOURCES)} (settings.toml poses)", param_hint="--poses"
        )
    return src


def poses_dir(cfg: Config, src: str) -> Path:
    """derived/refine/ (`scanmodel refine`) or derived/poses/ (the pose graph's): same files, same format."""
    return cfg.stage_dir("refine" if src == "refine" else "poses")


def _with(src: str) -> dict[str, str]:
    """Stage params for the pose source: none for posegraph, so its keys are the ones from before P7."""
    return {"poses": src} if src == "refine" else {}


def pose_source(poses_dir: Path) -> str:
    """Where a poses folder's poses come from: "refine" (`scanmodel refine`, bundle-adjusted), "arkit-posegraph"
    (`scanmodel poses` with drift correction) or "arkit" (without)."""
    if (poses_dir / "refine.json").is_file():
        return "refine"
    m = poses_dir / "manifest.json"
    graph = json.loads(m.read_text()).get("params", {}).get("pose_graph", True) if m.is_file() else True
    return "arkit-posegraph" if graph else "arkit"


def fused_from(cfg: Config) -> tuple[str, str]:
    """(pose source, its stamp) of derived/fuse/: the poses that structure, build and splat must use with it."""
    m = cfg.stage_dir("fuse") / "manifest.json"
    if not m.is_file():
        raise typer.BadParameter(
            "no fused points: run `scanmodel fuse` (or `scanmodel all`) first", param_hint="--poses"
        )
    stamp = json.loads(m.read_text()).get("summary", {}).get("pose_source") or "arkit-posegraph"
    return ("refine" if stamp == "refine" else "posegraph"), stamp


def _need_poses(cfg: Config, src: str) -> Path:
    pdir = poses_dir(cfg, src)
    if not (pdir / "poses.json").is_file():
        step = "refine" if src == "refine" else "poses"
        raise typer.BadParameter(
            f"{pdir} has no poses.json: run `scanmodel {step}` first", param_hint="--poses"
        )
    return pdir


def _fuse_stage(
    cfg: Config, src: str, voxel: float = 0.01, min_confidence: int = 1, color: bool = True
) -> Stage:
    params = {"voxel": voxel, "min_confidence": min_confidence, "color": color}
    pdir, out = poses_dir(cfg, src), cfg.stage_dir("fuse")
    return Stage(
        "fuse",
        inputs=lambda: [pdir / "poses.json", pdir / "frames.jsonl"],
        run=lambda: {**fuse.run(pdir, out, **params), "pose_source": pose_source(pdir)},  # type: ignore[arg-type]
        root=cfg.root,
        params={**params, **_with(src)},
        code=["stages/fuse.py", "points.py"],
    )


def _structure_stage(cfg: Config, src: str) -> Stage:
    pdir = poses_dir(cfg, src)
    return Stage(
        "structure",
        inputs=lambda: [
            pdir / "poses.json",
            cfg.stage_dir("fuse") / "points.ply",
            pdir / WALLS_DIR,
            _overrides(cfg),
            *[d / "room.json" for d in _ingested(cfg)],
        ],
        run=lambda: structure.run(
            pdir,
            cfg.stage_dir("fuse"),
            _overrides(cfg),
            cfg.scene_dir / "model.json",
        ),
        root=cfg.root,
        params={**structure.Params().__dict__, **_with(src)},
        code=["stages/structure.py", "points.py", "roomplan.py", "rooms.py", "boxes.py", "ownership.py"],
    )


def _eval_stage(cfg: Config, src: str) -> Stage:
    pdir = poses_dir(cfg, src)  # for the floors per session: the poses the points were fused from
    return Stage(
        "eval",
        inputs=lambda: [
            cfg.scene_dir / "model.json",
            cfg.stage_dir("fuse") / "points.ply",
            pdir / "frames.jsonl",
        ],
        run=lambda: eval_stage.run(
            cfg.scene_dir / "model.json", cfg.stage_dir("fuse"), cfg.stage_dir("eval"), pdir
        ),
        root=cfg.root,
        params=_with(src),
        code=["stages/eval.py", "points.py", "boxes.py", "consistency.py"],
    )


def _build_stage(cfg: Config, src: str) -> Stage:
    script = cfg.root / "blender" / "build.py"
    out = cfg.stage_dir("build")
    stills = poses_dir(cfg, src) / "stills.jsonl"

    def go() -> dict[str, object]:
        if not cfg.blender.is_file():
            raise FileNotFoundError(
                f"Blender not found at {cfg.blender}; set [tools] blender in settings.toml"
            )
        args = [
            "--scene",
            str(cfg.scene_dir / "model.json"),
            "--out",
            str(out / "model.blend"),
            "--render-preview",
            str(out / "preview.png"),
        ]
        if stills.is_file():
            args += ["--stills", str(stills)]
        summary: dict[str, object] = {}
        dense = cfg.stage_dir("fuse") / "dense.ply"
        if dense.is_file():  # the scan itself, as a reference layer and for the stairs
            summary = scanmesh.prepare(dense, cfg.scene_dir / "model.json", out)
            args += ["--scan", str(out / "scan.ply")]
            if (out / "stairs.ply").is_file():
                args += ["--stairs-mesh", str(out / "stairs.ply")]
        subprocess.run(
            [str(cfg.blender), "-b", "--factory-startup", "--python", str(script), "--", *args], check=True
        )
        return {"blend": cfg.rel(out / "model.blend"), **summary}

    return Stage(
        "build",
        inputs=lambda: [
            cfg.scene_dir / "model.json",
            stills,
            cfg.stage_dir("fuse") / "dense.ply",
            cfg.root / "blender",
        ],
        run=go,
        root=cfg.root,
        params=_with(src),
        code=["stages/scanmesh.py", "boxes.py"],
    )


def _run_refining(stage: Stage, out: Path, force: bool) -> None:
    """run_stage, with refine's "can't run here" as a clear error instead of a traceback."""
    try:
        run_stage(stage, out, force)
    except refine.RefineError as e:
        console.print(f"[red]{e}[/red]", highlight=False, soft_wrap=True)
        raise typer.Exit(1) from e


@app.command()
def schema(project: Project = None) -> None:
    """Regenerate the JSON Schemas from the Pydantic models."""
    for p in write_schemas(default_root()):
        console.print(f"wrote {p}")


@app.command("ingest")
def ingest_cmd(
    force: Force = False,
    from_: Annotated[str | None, typer.Option("--from", help="Import another format first: 'stray'.")] = None,
    src: Annotated[Path | None, typer.Argument(help="Capture to import (with --from).")] = None,
    project: Project = None,
    session: Annotated[str | None, typer.Option(help="Session id for the imported capture.")] = None,
    raw: Raw = None,
    skip: Skip = None,
) -> None:
    """Check the bundles in data/<project>/raw or --raw (importing a capture first with --from)."""
    cfg = _cfg()
    if from_ is not None:
        console.print(_project_line(cfg), highlight=False)
        if from_ != "stray" or src is None:
            raise typer.BadParameter("usage: scanmodel ingest --from stray <capture dir>")
        dst = import_stray(src.expanduser().resolve(), cfg.raw, cfg.project, session)
        console.print(f"imported {src} -> {dst}")
    run_stage(_ingest_stage(cfg, raw, skip), cfg.stage_dir("ingest"), force)


@app.command("poses")
def poses_cmd(
    force: Force = False,
    pose_graph: PoseGraph = None,
    stability: Annotated[
        bool,
        typer.Option(
            help="Also re-solve the pose graph without each session pair's closures and report how far poses"
            " move (leave_one_pair_out_max_m; about a third more time). A tuning diagnostic: off in `all` and `model`."
        ),
    ] = False,
    project: Project = None,
) -> None:
    """Merge sessions and fix the scene frame (gravity, yaw, front-door origin); optionally correct drift."""
    cfg = _cfg()
    run_stage(_poses_stage(cfg, pose_graph, stability), cfg.stage_dir("poses"), force)


@app.command("refine")
def refine_cmd(
    project: Project = None,
    force: Force = False,
    settle_drop_s: Annotated[
        float,
        typer.Option(help="Seconds left out after a session's unsettled start (ARKit's world still moving)."),
    ] = refine.SETTLE_DROP_S,
    readout_s: Annotated[
        float, typer.Option(help="Rolling-shutter readout to correct keypoints for (0: off; scan 4: 0.0147).")
    ] = 0.0,
    floor_sigma: Annotated[
        float, typer.Option(help="Floor-height observations' sigma in metres (0: no floor factors).")
    ] = refine.rf.FLOOR_SIGMA,
) -> None:
    """Bundle-adjust frames and stills on features matched across passes -> derived/refine/ (what every later
    stage reads, unless --poses posegraph)."""
    cfg = _cfg()
    _run_refining(_refine_stage(cfg, settle_drop_s, readout_s, floor_sigma), cfg.stage_dir("refine"), force)


@app.command("fuse")
def fuse_cmd(
    project: Project = None,
    force: Force = False,
    voxel: float = 0.01,
    min_confidence: int = 1,
    color: Annotated[bool, typer.Option(help="Colour points from video.mov.")] = True,
    poses_from: PosesFrom = None,
) -> None:
    """TSDF-fuse LiDAR depth into dense.ply and points.ply (derived/fuse/, stamped with its pose source)."""
    cfg = _cfg()
    src = _source(cfg, poses_from)
    _need_poses(cfg, src)
    run_stage(_fuse_stage(cfg, src, voxel, min_confidence, color), cfg.stage_dir("fuse"), force)


@app.command("structure")
def structure_cmd(force: Force = False, project: Project = None) -> None:
    """RoomPlan + plane refinement + overrides -> the project's scene/model.json (with the poses the points were
    fused from)."""
    cfg = _cfg()
    run_stage(_structure_stage(cfg, fused_from(cfg)[0]), cfg.stage_dir("structure"), force)


@app.command("eval")
def eval_cmd(force: Force = False, project: Project = None) -> None:
    """Geometric error of the project's scene/model.json against the fused points -> report.html."""
    cfg = _cfg()
    run_stage(_eval_stage(cfg, fused_from(cfg)[0]), cfg.stage_dir("eval"), force)
    console.print(_eval_line(cfg))
    console.print(f"report: {cfg.stage_dir('eval') / 'report.html'}")


@app.command("splat")
def splat_cmd(
    project: Project = None,
    force: Force = False,
    train: Annotated[bool, typer.Option(help="Train splats after exporting.")] = False,
    trainer: Annotated[
        str,
        typer.Option(
            help=f"{', '.join(splat.TRAINERS)} (paths: settings.toml [tools]; gsplat runs on Modal through trainer/)."
        ),
    ] = splat.DEFAULT_TRAINER,
    name: Annotated[
        str | None,
        typer.Option(help="Run name: trains into data/<project>/splat/runs/<name>/ (default: the trainer)."),
    ] = None,
    iterations: Annotated[int, typer.Option("--iterations", "-n", help="Training steps.")] = 30000,
    downscale: Annotated[
        float | None,
        typer.Option(help="Train at 1/downscale resolution (default: msplat fits memory, Brush full)."),
    ] = None,
    eval_every: Annotated[
        int, typer.Option(help="msplat/Brush: hold out every nth view for PSNR/SSIM (0: none).")
    ] = 8,
    select: Annotated[
        str,
        typer.Option(
            help="Video frames: motion (a slot per --min-move / --min-turn / --max-gap-s, its sharpest frame,"
            " blurry ones dropped) or time (the first frame of every --every-s)."
        ),
    ] = "motion",
    every_s: Annotated[float, typer.Option(help="--select time: seconds between frames.")] = 0.5,
    min_move: Annotated[float, typer.Option(help="--select motion: metres moved per slot.")] = 0.12,
    min_turn: Annotated[float, typer.Option(help="--select motion: degrees turned per slot.")] = 6.0,
    max_gap_s: Annotated[float, typer.Option(help="--select motion: seconds per slot at most.")] = 1.0,
    rel_blur: Annotated[
        float,
        typer.Option(help="Drop video frames and stills below this x their neighbours' sharpness."),
    ] = 0.6,
    blur_ref: Annotated[
        str,
        typer.Option(
            help="--select motion: a frame is blurry against the median of all frames within ±1 s"
            " (neighbours) or of the picked frames within ±2 s (picked)."
        ),
    ] = "neighbours",
    max_side: int = 1920,
    extrap_session: Annotated[
        str,
        typer.Option(
            help="Session left out of training as the extrap test set: auto (the *-lap one), none, or an id."
        ),
    ] = "auto",
    trainer_args: Annotated[
        list[str] | None, typer.Option("--trainer-arg", help="Extra trainer argument (repeatable).")
    ] = None,
    prune: Annotated[
        bool, typer.Option(help="msplat: remove splats in LiDAR free space during training (floaters).")
    ] = True,
    deband: Annotated[
        str,
        typer.Option(
            help="Divide LED flicker bands out of frames and stills: auto (sessions found banded, a cheap"
            " check), on, off."
        ),
    ] = "auto",
    mains_hz: Annotated[
        float, typer.Option(help="--deband: mains frequency (lights flicker at twice it).")
    ] = 50.0,
    poses_from: PosesFrom = None,
    dataset: Annotated[
        str | None,
        typer.Option(help="Dataset folder under splat/datasets/ (export and --train; default: default)."),
    ] = None,
    test_poses: Annotated[
        bool,
        typer.Option(help="Re-solve the test frames' poses against the training frames (test_poses.json)."),
    ] = True,
) -> None:
    """Export a nerfstudio dataset (stills, video frames, depth, split, mesh) and optionally train it.

    The initial points and mesh priors are derived/fuse/'s, which must be fused from the same poses."""
    cfg = _cfg()
    out = cfg.stage_dir("splat")
    if select not in splat.SELECT_MODES:
        raise typer.BadParameter(f"pick one of {', '.join(splat.SELECT_MODES)}", param_hint="--select")
    if blur_ref not in splat.BLUR_REFS:
        raise typer.BadParameter(f"pick one of {', '.join(splat.BLUR_REFS)}", param_hint="--blur-ref")
    if deband not in DEBAND_MODES:
        raise typer.BadParameter(f"pick one of {', '.join(DEBAND_MODES)}", param_hint="--deband")
    dataset = dataset or "default"
    if not dataset or "/" in dataset or dataset.startswith(".") or dataset in splat.TRAINERS:
        raise typer.BadParameter("a plain folder name, not a trainer's", param_hint="--dataset")
    src = _source(cfg, poses_from)
    pdir = _need_poses(cfg, src)
    fdir = cfg.stage_dir("fuse")
    fused, mesh_source = fused_from(cfg)
    if not (fdir / "dense.ply").is_file():
        raise typer.BadParameter(f"{fdir} has no dense.ply: run `scanmodel fuse` first", param_hint="--poses")
    if fused != src:
        raise typer.BadParameter(
            f"{fdir} holds points fused from {mesh_source} poses, not {src}: run `scanmodel fuse --poses {src}`"
            f" (or `scanmodel all --poses {src}`) first",
            param_hint="--poses",
        )
    data_dir = cfg.dataset_dir(dataset)
    params = {
        "every_s": every_s,
        "max_side": max_side,
        "extrap_session": extrap_session,
        "select": select,
        "min_move": min_move,
        "min_turn": min_turn,
        "max_gap_s": max_gap_s,
        "rel_blur": rel_blur,
        "blur_ref": blur_ref,
        "deband": deband,
        "mains_hz": mains_hz,
        "test_poses": test_poses,
        **_with(src),
    }
    stage = Stage(
        f"splat {dataset}",
        inputs=lambda: _splat_inputs(cfg, src),
        run=lambda: splat.export(
            pdir,
            fdir,
            data_dir,
            every_s=every_s,
            max_side=max_side,
            extrap_session=extrap_session,
            select=select,
            min_move=min_move,
            min_turn=min_turn,
            max_gap_s=max_gap_s,
            rel_blur=rel_blur,
            blur_ref=blur_ref,
            sharpness_dir=out / "analysis" / "sharpness",
            deband=deband,
            mains_hz=mains_hz,
            deband_dir=out / "analysis" / "deband",
            pose_source=pose_source(pdir),
            mesh_pose_source=mesh_source,
            test_poses=test_poses,
        ),
        root=cfg.root,
        params=params,
        code=SPLAT_CODE,
    )
    run_stage(stage, data_dir, force)
    if not train:
        return
    if trainer not in splat.TRAINERS:
        raise typer.BadParameter(f"pick one of {', '.join(splat.TRAINERS)}", param_hint="--trainer")
    run = name or trainer
    if run in ("dataset", dataset, "exports") or "/" in run or run.startswith("."):
        raise typer.BadParameter("a plain folder name other than the dataset's", param_hint="--name")
    exe = {"msplat": cfg.msplat, "brush": cfg.brush, "gsplat": "uv"}[trainer]
    extra = trainer_args or []
    train_params = {
        "trainer": trainer,
        "iterations": iterations,
        "downscale": downscale,
        "eval_every": eval_every,
        "extra": extra,
        "prune": prune and trainer == "msplat",
    }
    train_stage = Stage(
        f"splat {run}",
        inputs=lambda: [data_dir / "transforms.json", data_dir / "points.ply"],
        run=lambda: splat.train(
            trainer,
            exe,
            data_dir,
            cfg.runs_dir / run,
            iterations,
            eval_every,
            downscale,
            extra,
            prune=bool(train_params["prune"]),
            name=run,
            project=cfg.project,
            root=cfg.root,
        ),
        root=cfg.root,
        params=train_params,
        code=TRAIN_CODE,
    )
    run_stage(train_stage, cfg.runs_dir / run, force)


SPLAT_CODE = ["stages/splat.py", "depthviews.py", "deband.py", "testposes.py"]  # a dataset export's
TRAIN_CODE = ["stages/splat.py", "msplat_train.py", "freespace.py", "depthviews.py"]  # a local training run's


def _splat_inputs(cfg: Config, src: str) -> list[Path]:
    pdir, fdir = poses_dir(cfg, src), cfg.stage_dir("fuse")
    return [
        pdir / "poses.json",
        pdir / "frames.jsonl",
        pdir / "stills.jsonl",
        fdir / "points.ply",
        fdir / "dense.ply",
    ]


def _export_stage(cfg: Config, run: Any = dict) -> Stage:
    blend = cfg.stage_dir("build") / "model.blend"
    return Stage("export", inputs=lambda: [blend, cfg.root / "blender" / "export.py"], run=run, root=cfg.root)


@app.command("test-poses")
def test_poses_cmd(
    project: Project = None,
    dataset: Annotated[str | None, typer.Option(help="Splat dataset name.")] = None,
    data: Annotated[
        Path | None, typer.Option(help="Any splat dataset directory instead (e.g. another scan's).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="File to write instead of the dataset's test_poses.json.")
    ] = None,
) -> None:
    """Re-solve a splat dataset's test poses against its training frames (test_poses.json), e.g. for a
    dataset exported before `scanmodel splat` did it. Leaves the dataset's other files (and its cloud hash) as
    they are."""
    d = data or _cfg().dataset_dir(dataset)
    if not (d / "transforms.json").is_file():
        raise typer.BadParameter(
            f"{d} has no transforms.json: run `scanmodel splat` first", param_hint="--data"
        )
    testposes.write(d, out)
    console.print(f"test poses: {out or d / testposes.FILE}")


view_app = typer.Typer(
    no_args_is_help=True, help="Named camera views for splat comparisons (project scene/views.json)."
)
app.add_typer(view_app, name="view")


@view_app.command("add")
def view_add(
    name: Annotated[str, typer.Argument(help="Short name, e.g. kitchen-door.")],
    project: Project = None,
    url: Annotated[
        str | None,
        typer.Argument(help="Viewer URL with ?pose= (or ?cam= and ?look=) and ?fov=, quoted."),
    ] = None,
    note: Annotated[str, typer.Option(help="What to look at in this view.")] = "",
    width: int = splat.VIEW_SIZE[0],
    height: int = splat.VIEW_SIZE[1],
    image: Annotated[
        str | None,
        typer.Option(
            help="Instead of a URL: a captured frame or still's pose, by id (<session>/342, <session>/s44)"
            " or dataset image stem (<session>_000342, <session>_s44). Height follows its aspect."
        ),
    ] = None,
    poses_from: PosesFrom = None,
) -> None:
    """Add a named view from a viewer URL or a captured image's pose (replaces a view of the same name)
    in data/<project>/scene/views.json."""
    cfg = _cfg()
    path = cfg.views_file
    if (url is None) == (image is None):
        raise typer.BadParameter("give a viewer URL or --image, not both", param_hint="URL")
    if image is not None:
        try:
            view = splat.view_from_image(
                name, splat.find_image(poses_dir(cfg, _source(cfg, poses_from)), image), note, (width, height)
            )
        except (KeyError, FileNotFoundError) as e:
            raise typer.BadParameter(str(e), param_hint="--image") from e
    else:
        assert url is not None
        try:
            view = splat.view_from_url(name, url, note, (width, height))
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="URL") from e
    replaced = splat.add_named_view(path, view)
    console.print(f"{'replaced' if replaced else 'added'} view {name} in {path}")


@view_app.command("list")
def view_list(project: Project = None) -> None:
    """List the named views with the viewer query that shows each."""
    path = _cfg().views_file
    for v in splat.read_named_views(path):
        console.print(f"[bold]{v['name']}[/bold]  {v['note']}", highlight=False)
        console.print(f"  {splat.view_url(v)}", markup=False, highlight=False, soft_wrap=True)


@app.command("meshcheck")
def meshcheck_cmd(
    project: Project = None,
    poses_from: Annotated[
        str | None,
        typer.Option(
            "--poses", help="Frames' poses: refine or posegraph (default: those the mesh was fused from)."
        ),
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(help="Only this splat dataset's video frames (a folder under splat/datasets/)."),
    ] = None,
    max_frames: int = 400,
    out: Annotated[Path | None, typer.Option(help="Write the result as JSON here.")] = None,
) -> None:
    """derived/fuse/'s mesh vs each frame's own LiDAR at its pose: depth residual and normal angle (the
    trainer's mesh priors against its LiDAR depth)."""
    from . import meshcheck

    cfg = _cfg()
    fused, mesh_source = fused_from(cfg)
    src = _source(cfg, poses_from or fused)
    ids = None
    if dataset is not None:
        d = cfg.dataset_dir(dataset)
        if not (d / "transforms.json").is_file():
            raise typer.BadParameter(f"{d} has no transforms.json", param_hint="--dataset")
        t = json.loads((d / "transforms.json").read_text())
        split = json.loads((d / "split.json").read_text()) if (d / "split.json").is_file() else {}
        ids = {
            f"{session}/{int(idx)}"
            for f in [*t["frames"], *split.get("extrap_frames", [])]
            if f.get("camera") == "video"
            for session, _, idx in [splat.stem(f).rpartition("_")]
        }
    r = meshcheck.mesh_vs_lidar(
        cfg.stage_dir("fuse") / "dense.ply", _need_poses(cfg, src), ids, max_frames=max_frames
    )
    r = {"mesh_pose_source": mesh_source, "poses_from": src, "dataset": dataset, **r}
    a = r["all"]
    assert isinstance(a, dict)
    console.print(
        f"mesh from {mesh_source} vs LiDAR at {src} poses, {r['frames']} frames: |residual| median"
        f" {a['abs_median_mm']} mm, p90 {a['abs_p90_mm']} mm, bias {a['bias_median_mm']} mm, within 1 cm"
        f" {a['within_1cm']:.1%}; normals median {a['normal_median_deg']}°, within 10° {a['normal_within_10deg']:.1%}",
        highlight=False,
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=1) + "\n")
        console.print(f"wrote {out}")


@app.command("build")
def build_cmd(force: Force = False, project: Project = None) -> None:
    """Build data/<project>/build/model.blend from data/<project>/scene/model.json with Blender (headless)."""
    cfg = _cfg()
    run_stage(_build_stage(cfg, fused_from(cfg)[0]), cfg.stage_dir("build"), force)


def publish_splats(
    splat_dir: Path,
    viewer: Path,
    default_trainer: str,
    url_base: str = "data/",
    dataset_dir: Path | None = None,
    root: Path | None = None,
) -> list[dict[str, object]]:
    """Link every trained splat into the viewer and describe them in `splats.json`.

    `viewer` is the folder the files go to (a scan's `data/scans/<id>/`), `url_base` the same folder as the
    page fetches it (`data/scans/<id>/`).

    A splat is any `*.ply` directly in a subdirectory of `splat_dir` that has a manifest.json (the
    trainer's run): `msplat/splat.ply` -> id `msplat`, a variant `msplat/splat.clean.ply` -> id
    `msplat-clean`. An optional `<stem>.json` next to a variant adds to its summary. `current` says
    whether the run's inputs are its dataset as it is now (else it was trained on an older one); a run's
    dataset is the one it recorded (`run_dataset`), else `dataset_dir`.

    Other links in `splats/` (added by hand: another scan's run, a run without a trainer manifest) are
    kept with their `splats.json` entries, unless a run here now has the same id or the file they point to
    is gone. The default is `default_trainer` if it exists, else the previous default if it is still
    there, else the first splat.
    """
    out = viewer / "splats"
    index = viewer / "splats.json"
    old = json.loads(index.read_text()) if index.is_file() else {}
    kept: dict[str, Path] = {}  # id -> the splat an existing link points to
    if out.is_dir():
        for link in sorted(out.glob("*.ply")):
            if link.is_symlink() and link.resolve().is_file():
                kept[link.stem] = link.resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    default_dataset = dataset_dir or splat_dir / "dataset"
    current: dict[Path, str] = {}  # dataset -> fingerprint of its files now

    def current_inputs(dataset: Path) -> str:
        if dataset not in current:
            current[dataset] = fingerprint([dataset / "transforms.json", dataset / "points.ply"], root)
        return current[dataset]

    entries: list[dict[str, object]] = []
    for run_dir in sorted(d for d in splat_dir.glob("*") if (d / "manifest.json").is_file()):
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if "trainer" not in manifest.get("params", {}):
            continue  # the dataset export, not a training run
        dataset = run_dataset(run_dir, manifest, root) or default_dataset
        for ply in sorted(run_dir.glob("*.ply")):
            variant = ply.stem.removeprefix("splat").lstrip(".-_")
            sid = f"{run_dir.name}-{variant}" if variant else run_dir.name
            summary = {k: v for k, v in manifest.get("summary", {}).items() if k != "command"}
            extra = ply.with_suffix(".json")
            if variant and extra.is_file():
                summary.update(json.loads(extra.read_text()))
            if variant:
                summary["splats"] = splat.ply_vertex_count(ply)
            (out / f"{sid}.ply").symlink_to(ply.resolve())  # hundreds of MB: linked, not copied
            entries.append({
                "id": sid,
                "label": f"{run_dir.name} ({variant})" if variant else run_dir.name,
                "url": f"{url_base}splats/{sid}.ply",
                "trainer": manifest["params"]["trainer"],
                "params": manifest["params"],
                "summary": summary,
                "trained_at": manifest.get("finished_at") if not variant else None,
                "file_at": datetime.fromtimestamp(ply.stat().st_mtime, UTC).isoformat(),
                "size_mb": round(ply.stat().st_size / 1e6, 1),
                "current": manifest.get("inputs") == current_inputs(dataset),
            })  # fmt: skip
    ids = {e["id"] for e in entries}
    for sid, target in kept.items():
        if sid not in ids:
            (out / f"{sid}.ply").symlink_to(target)
    entries += [e for e in old.get("splats", []) if e.get("id") in kept.keys() - ids]
    ids = [e["id"] for e in entries]
    default = next((d for d in (default_trainer, old.get("default")) if d in ids), ids[0] if ids else None)
    index.write_text(json.dumps({"default": default, "splats": entries}, indent=1) + "\n")
    return entries


def run_dataset(run_dir: Path, manifest: Mapping[str, Any], root: Path | None = None) -> Path | None:
    """The dataset a training run used: its manifest summary's `dataset` (`scanmodel splat --train`), the cloud
    run's `run.json` `data_dir`, or a dataset folder in the recorded command. None if it recorded none."""
    summary = manifest.get("summary") or {}
    candidates = [summary.get("dataset")]
    if (run_dir / "run.json").is_file():
        candidates.append(json.loads((run_dir / "run.json").read_text()).get("data_dir"))
    candidates += [c for c in summary.get("command") or [] if isinstance(c, str) and not c.startswith("-")]
    for c in candidates:
        if isinstance(c, str) and c and (from_repo(c, root) / "transforms.json").is_file():
            return from_repo(c, root)
    return None


SCANS_INDEX = "scans.json"  # in viewer/public/data/: the exported scans and the default one


def viewer_data_dir(root: Path) -> Path:
    """viewer/public/data of the repo: all scans share one."""
    return root / "viewer" / "public" / "data"


def scan_label(cfg: Config, sid: str) -> str:
    """`third · 2026-10-07 · conservatory, living, ...` from the sessions' names (`YYYYMMDD-HHMMSS-room`)."""
    p = cfg.stage_dir("poses") / "poses.json"
    sessions = [s["session"] for s in json.loads(p.read_text()).get("sessions", [])] if p.is_file() else []
    parts = [s.split("-", 2) for s in sessions]
    rooms = [x[2] for x in parts if len(x) == 3]
    day = parts[0][0] if parts and len(parts[0][0]) == 8 and parts[0][0].isdigit() else ""
    date = f"{day[:4]}-{day[4:6]}-{day[6:]}" if day else ""
    return " · ".join(x for x in (sid, date, ", ".join(dict.fromkeys(rooms))) if x)


def publish_views(views_file: Path, dst: Path) -> int:
    """The project's named views (data/<project>/scene/views.json) as the viewer takes them: `views.json` with each view's
    `pose` (row-major camera-to-world, OpenGL axes, as `?pose=`) and `fov`. Returns how many."""
    views = splat.read_named_views(views_file)
    rows = []
    for v in views:
        gl = np.asarray(v["c2w"], dtype=float) @ GL_CV
        rows.append({"name": v["name"], "note": v.get("note", ""), "fov": v["fov_y_deg"],
                     "pose": [round(float(x), 6) + 0.0 for x in gl.ravel()]})  # fmt: skip
    dst.write_text(json.dumps({"views": rows}, indent=1) + "\n")
    return len(rows)


def register_scan(viewer: Path, entry: Mapping[str, object], make_default: bool) -> dict[str, Any]:
    """Add or replace `entry` in viewer/public/data/scans.json. The default scan stays as it is unless
    `make_default` (or there was none yet, or it is gone). Returns the index."""
    index = viewer / SCANS_INDEX
    old = json.loads(index.read_text()) if index.is_file() else {}
    scans = [s for s in old.get("scans", []) if s.get("id") != entry["id"]]
    scans = sorted([*scans, entry], key=lambda s: str(s["id"]))
    ids = [s["id"] for s in scans]
    default = entry["id"] if make_default or old.get("default") not in ids else old["default"]
    out = {"default": default, "scans": scans}
    index.write_text(json.dumps(out, indent=1) + "\n")
    return out


def _glb_stage(cfg: Config) -> Stage:
    """model.blend's structure as data/<project>/export/model.glb (headless Blender, blender/export.py)."""
    blend, out = cfg.stage_dir("build") / "model.blend", cfg.stage_dir("export")

    def go() -> dict[str, object]:
        if not cfg.blender.is_file():
            raise FileNotFoundError(
                f"Blender not found at {cfg.blender}; set [tools] blender in settings.toml"
            )
        if not blend.is_file():
            raise FileNotFoundError(f"{blend} missing: run `scanmodel build` first")
        subprocess.run(
            [str(cfg.blender), "-b", "--factory-startup", "--python", str(cfg.root / "blender" / "export.py"),
             "--", "--blend", str(blend), "--out", str(out / "model.glb")],
            check=True,
        )  # fmt: skip
        return {"glb": cfg.rel(out / "model.glb")}

    return _export_stage(cfg, go)


@app.command("export")
def export_cmd(
    project: Project = None,
    force: Force = False,
    trainer: Annotated[
        str, typer.Option(help="Splat the viewer selects first (msplat, brush, ...).")
    ] = "msplat",
    scan: Annotated[
        str | None,
        typer.Option(help="The scan's id in the viewer (default: the project, e.g. fifth)."),
    ] = None,
    viewer_default: Annotated[
        bool, typer.Option("--viewer-default", help="Make this scan the one the viewer opens without ?scan=.")
    ] = False,
) -> None:
    """Data for the web viewer: model.glb (structure, from the .blend), every trained splat and the named views
    of this project's scan, in viewer/public/data/scans/<scan>/; scans.json lists the scans. Other scans'
    folders are left alone, so each project adds its own."""
    cfg = _cfg()
    out = cfg.stage_dir("export")
    sid = scan or cfg.project
    data = viewer_data_dir(cfg.root)
    viewer = data / "scans" / sid
    base = f"data/scans/{sid}/"

    run_stage(_glb_stage(cfg), out, force)
    # The rest runs every time (copies and links): a new scan folder, splat or named view shows up without --force.
    viewer.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / "model.glb", viewer / "model.glb")
    entries = publish_splats(cfg.runs_dir, viewer, trainer, base, cfg.dataset_dir(), cfg.root)
    n_views = publish_views(cfg.views_file, viewer / "views.json")
    for e in entries:
        stale = "" if e["current"] else "  [yellow](older dataset)[/yellow]"
        console.print(f"  splat {e['id']}: {e['size_mb']} MB{stale}", highlight=False)
    if not entries:
        console.print(
            "[yellow]no trained splat: run `scanmodel splat --train` (viewer shows the structure only)[/yellow]"
        )
    splats_index = json.loads((viewer / "splats.json").read_text())
    index = register_scan(data, {
        "id": sid,
        "label": scan_label(cfg, sid),
        "base": base,
        "default_splat": splats_index.get("default"),
        "splats": len(entries),
        "views": n_views,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }, viewer_default)  # fmt: skip
    console.print(
        f"scan {sid}: {len(entries)} splats, {n_views} named views in {viewer}; viewer default scan:"
        f" {index['default']} (of {', '.join(str(s['id']) for s in index['scans'])})",
        highlight=False,
    )
    query = "" if index["default"] == sid else f"/?scan={sid}"
    console.print(f"viewer: just viewer  ->  http://localhost:5173{query}")


def model_stages(
    cfg: Config,
    src: str,
    raw: Path | None = None,
    skip: list[str] | None = None,
    pose_graph: bool | None = None,
    build: bool = True,
) -> list[Stage]:
    """The stages from a scan to the Blender model, in order: what `scanmodel all` and `scanmodel model` run.
    Nothing of the splat route (`scanmodel splat`) is in it."""
    return [
        _ingest_stage(cfg, raw, skip),
        _poses_stage(cfg, pose_graph),
        *([_refine_stage(cfg)] if src == "refine" else []),
        _fuse_stage(cfg, src),
        _structure_stage(cfg, src),
        _eval_stage(cfg, src),
        *([_build_stage(cfg, src)] if build else []),
    ]


def run_stages(stages: list[Stage], cfg: Config, force: bool = False) -> list[tuple[str, bool, float]]:
    """Run each stage unless it is up to date, with an `[n/N] stage` header; then a table of what ran and
    how long it took. Returns (stage, ran, seconds) per stage. Stops at the first stage that fails."""
    import time

    done: list[tuple[str, bool, float]] = []
    try:
        for n, st in enumerate(stages, 1):
            console.rule(f"[{n}/{len(stages)}] {st.name}", align="left", style="dim")
            t = time.perf_counter()
            try:
                ran = run_stage(st, cfg.stage_dir(st.name), force)
            except refine.RefineError as e:
                console.print(f"[red]{e}[/red]", highlight=False, soft_wrap=True)
                raise typer.Exit(1) from e
            except BaseException as e:
                what = "stopped" if isinstance(e, KeyboardInterrupt) else "failed"
                console.print(
                    f"[red]{st.name} {what}[/red]: the stages before it are kept, a re-run resumes here"
                )
                raise
            done.append((st.name, ran, time.perf_counter() - t))
    finally:
        if done:
            console.print(stage_table(done))
    return done


def stage_table(done: list[tuple[str, bool, float]]) -> str:
    """`ingest 41.2 s ran`, one line per stage, and the total."""

    def dur(s: float) -> str:
        return f"{s:6.1f} s" if s < 120 else f"{int(s // 60):3d} min {int(s % 60):02d} s"

    rows = [f"  {name:10} {dur(sec):>12}  {'ran' if ran else 'cached'}" for name, ran, sec in done]
    return "\n".join(["stages:", *rows, f"  {'total':10} {dur(sum(s for _, _, s in done)):>12}"])


@app.command("all")
def all_cmd(
    project: Project = None,
    force: Force = False,
    build: Annotated[bool, typer.Option(help="Also build the .blend.")] = True,
    raw: Raw = None,
    skip: Skip = None,
    pose_graph: PoseGraph = None,
    poses_from: PosesFrom = None,
) -> None:
    """ingest -> poses -> refine -> fuse -> structure -> eval (-> build). Runs only what changed.

    `--poses posegraph` leaves refine out and runs the rest on the pose graph's poses, as before P7. If refine
    can't run (no video, too few tracks) this stops with that message: it never falls back on its own."""
    cfg = _cfg()
    src = _source(cfg, poses_from)
    run_stages(model_stages(cfg, src, raw, skip, pose_graph, build), cfg, force)
    console.print(_eval_line(cfg))


def _open(path: Path) -> None:
    """Open a file with its app (Blender for a .blend, the browser for a report)."""
    if not path.exists():
        raise typer.BadParameter(f"{path} does not exist yet", param_hint="WHAT")
    opener = "open" if shutil.which("open") else "xdg-open"
    subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@app.command("model")
def model_cmd(
    scan: Annotated[
        str | None,
        typer.Argument(
            help="The scan: a project folder under data/ (e.g. fifth). Default: --project, else settings.toml"
            " project."
        ),
    ] = None,
    glb: Annotated[bool, typer.Option("--glb", help="Also export model.glb (structure only, glTF).")] = False,
    open_: Annotated[bool, typer.Option("--open", help="Open the .blend in Blender when done.")] = False,
    force: Force = False,
    raw: Raw = None,
    skip: Skip = None,
    poses_from: Annotated[
        str | None,
        typer.Option(
            "--poses",
            help="posegraph: leave refine (bundle adjustment, ~8 min on a 15 min scan) out. Faster, but"
            " walls are worse (scan 3: p95 87 vs 60 mm), and switching back re-runs fuse and on.",
        ),
    ] = None,
    project: Project = None,
) -> None:
    """A Blender model from a scan, nothing else: ingest -> poses -> refine -> fuse -> structure -> eval ->
    build in data/<project>/. Only what changed runs again. Splats are separate (`scanmodel splat`, `just gs`)."""
    try:
        cfg = load_config(project=scan or _cli_project())
    except ConfigError as e:
        raise typer.BadParameter(str(e), param_hint="SCAN") from e
    src_raw = raw.expanduser().resolve() if raw else cfg.raw
    if not find_bundles(src_raw):
        known = sorted(d.name for d in (cfg.root / "data").glob("*") if find_bundles(d / "raw"))
        hint = f"; projects with scans: {', '.join(known)}" if known else ""
        raise typer.BadParameter(
            f"{src_raw} has no session bundles (<session>/session.json){hint}", param_hint="SCAN"
        )
    console.print(_project_line(cfg), highlight=False)
    console.print(
        f"scan      {src_raw}  ({len(find_bundles(src_raw, skip or []))} sessions)", highlight=False
    )
    src = _source(cfg, poses_from)
    stages = model_stages(cfg, src, raw, skip, build=True)
    if glb:
        stages.append(_glb_stage(cfg))
    run_stages(stages, cfg, force)
    console.print(_eval_line(cfg))
    blend = cfg.stage_dir("build") / "model.blend"
    console.print(f"model   {blend}", highlight=False)
    console.print(f"plan    {cfg.stage_dir('build') / 'preview.png'}", highlight=False)
    console.print(f"report  {cfg.stage_dir('eval') / 'report.html'}", highlight=False)
    if glb:
        console.print(f"glb     {cfg.stage_dir('export') / 'model.glb'}", highlight=False)
    if open_:
        _open(blend)
    else:
        console.print(
            f"open it: just open blend --project {cfg.project}  (or: open {blend})", highlight=False
        )


OPENABLE = {
    "blend": ("build", "model.blend"),
    "plan": ("build", "preview.png"),
    "report": ("eval", "report.html"),
    "glb": ("export", "model.glb"),
}


@app.command("open")
def open_cmd(
    what: Annotated[str, typer.Argument(help=f"{', '.join(OPENABLE)} or data.")] = "blend",
    project: Project = None,
) -> None:
    """Open the project's model (model.blend in Blender), plan render, eval report, model.glb or its data
    folder."""
    cfg = _cfg()
    if what == "data":
        _open(cfg.data)
        return
    if what not in OPENABLE:
        raise typer.BadParameter(f"pick one of {', '.join([*OPENABLE, 'data'])}", param_hint="WHAT")
    stage, name = OPENABLE[what]
    path = cfg.stage_dir(stage) / name
    console.print(f"opening {path}", highlight=False)
    _open(path)


def _floor_note(stage: str, summary: dict[str, object]) -> str:
    """Floor consistency in `scanmodel status`: refine's session floor offsets, eval's per-room spread."""
    if stage == "refine" and isinstance(off := summary.get("floor_session_offset_max_mm"), dict):
        plane = summary.get("floor_camera_dz_plane") or {}
        return (
            f"  floors: sessions apart up to {off.get('after')} mm (input {off.get('before')} mm),"  # type: ignore[union-attr]
            f" height change {plane.get('x_mm_per_m')} / {plane.get('y_mm_per_m')} mm/m along x / y"  # type: ignore[union-attr]
        )
    if stage == "eval" and summary.get("spread_max_m") is not None:
        return f"  floors: {_floor_text(summary)}"
    return ""


def _floor_text(s: dict[str, object]) -> str:
    def cm(v: object) -> str:
        return f"{float(v) * 100:.1f} cm" if isinstance(v, int | float) else "-"

    return (
        f"spread across sessions up to {cm(s.get('spread_max_m'))} ({s.get('spread_max_room')}),"
        f" level rooms off by up to {cm(s.get('level_offset_max_m'))} ({s.get('level_offset_max_room')})"
    )


def _eval_line(cfg: Config) -> str:
    m = cfg.stage_dir("eval") / "manifest.json"
    if not m.is_file():
        return ""
    s = json.loads(m.read_text()).get("summary", {})

    def mm(v: object) -> str:
        return f"{float(v) * 1000:.1f}" if isinstance(v, int | float) else "-"

    floors = f"; floors: {_floor_text(s)}" if s.get("spread_max_m") is not None else ""
    warn = f"\n[yellow]warning[/yellow]: {s['split_warning_line']}" if s.get("split_warning_line") else ""
    if s.get("observed_p95_m") is None:  # a run from before seen and unseen faces were told apart
        return f"eval: walls mean {mm(s.get('mean_m'))} mm, p95 {mm(s.get('p95_m'))} mm{floors}{warn}"
    return (
        f"eval: seen wall faces mean {mm(s.get('observed_mean_m'))} mm, p95 {mm(s.get('observed_p95_m'))} mm"
        f" ({s.get('observed_faces')} of {s.get('observed_faces_total')} measured; unseen {mm(s.get('unobserved_mean_m'))} / {mm(s.get('unobserved_p95_m'))}, all faces"
        f" {mm(s.get('mean_m'))} / {mm(s.get('p95_m'))}); {s.get('faces_gross')} faces 10-50 cm off{floors}{warn}"
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def bench(ctx: typer.Context, project: Project = None) -> None:
    """Benchmark on ARKitScenes ground truth and data/<project>/checks.json (see scanmodel.bench). Without
    --project: every project with a checks.json."""
    from . import bench as b

    selected = _cli_project()
    raise typer.Exit(b.main([*ctx.args, *(["--project", selected] if selected else [])]))


@app.command()
def status(project: Project = None) -> None:
    """Show config and which stages have run."""
    cfg = _cfg()
    console.print(_project_line(cfg), highlight=False)
    console.print(
        f"root      {cfg.root}\ndata      {cfg.data}\nraw       {cfg.raw}\nscene     {cfg.scene_dir}\n"
        f"blender   {cfg.blender} ({'ok' if cfg.blender.is_file() else 'missing'})\n"
        f"brush     {cfg.brush} ({'ok' if shutil.which(cfg.brush) else 'missing'})\n"
        f"msplat    {cfg.msplat} ({'ok' if shutil.which(os.path.expanduser(cfg.msplat)) else 'missing'})"
    )
    console.print(f"bundles   {len(find_bundles(cfg.raw))} in {cfg.raw}")
    console.print(f"poses     {cfg.poses} (settings.toml poses; --poses)")
    for name in ("ingest", "poses", "refine", "fuse", "structure", "eval", "build", "export", "splat"):
        m = (cfg.dataset_dir() if name == "splat" else cfg.stage_dir(name)) / "manifest.json"
        record = json.loads(m.read_text()) if m.is_file() else {}
        summary = record.get("summary", {})
        stamp = summary.get("pose_source")
        took = ""
        if record.get("started_at") and record.get("finished_at"):
            sec = (
                datetime.fromisoformat(record["finished_at"]) - datetime.fromisoformat(record["started_at"])
            ).total_seconds()
            took = f" {sec:6.0f} s, {record['finished_at'][:16].replace('T', ' ')}"
        console.print(
            f"{name:10}{'done' if m.is_file() else '-'}{took}{f'  ({stamp})' if stamp else ''}{_floor_note(name, summary)}",
            highlight=False,
        )
    for name in splat.TRAINERS:
        m = cfg.runs_dir / name / "manifest.json"
        if m.is_file():
            r = json.loads(m.read_text())["summary"]
            metrics = "".join(f", {k.upper()} {r[k]}" for k in ("psnr", "ssim") if k in r)
            console.print(
                f"  {name:8}{r['splats']:,} splats, {r['iterations']} steps, {r['wall_s']} s{metrics}"
            )


def _rekey_inputs(cfg: Config, manifest: Path) -> tuple[list[Path], list[str]] | None:
    """(inputs, code modules) a project's stage manifest is keyed on, as the stage builders define them."""
    m = json.loads(manifest.read_text())
    params = m.get("params") or {}
    src = "refine" if params.get("poses") == "refine" else "posegraph"
    folder = manifest.parent
    core = {
        "ingest": lambda: _ingest_stage(
            cfg, cfg.from_repo(params["raw"]) if params.get("raw") else None, params.get("skip")
        ),
        "poses": lambda: _poses_stage(cfg),
        "refine": lambda: _refine_stage(cfg),
        "fuse": lambda: _fuse_stage(cfg, src),
        "structure": lambda: _structure_stage(cfg, src),
        "eval": lambda: _eval_stage(cfg, src),
        "build": lambda: _build_stage(cfg, src),
        "export": lambda: _export_stage(cfg),
    }
    if folder.name in core and folder == cfg.stage_dir(folder.name):
        st = core[folder.name]()
        return st.inputs(), st.code
    if folder.parent == cfg.stage_dir("splat") / "datasets":
        return _splat_inputs(cfg, src), SPLAT_CODE
    if folder.parent == cfg.runs_dir and "trainer" in params:
        d = run_dataset(folder, m, cfg.root) or cfg.dataset_dir()
        # gsplat's code hash is the trainer's (git commit): left as it is
        return [d / "transforms.json", d / "points.ply"], [] if params["trainer"] == "gsplat" else TRAIN_CODE
    return None


@app.command()
def rekey(
    project: Project = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Only list what would change.")] = False,
) -> None:
    """Mark every stage of the project that has a manifest as up to date with its files and code as they are
    now, without running it (each keeps its recorded params). Only after a change known not to alter
    results, e.g. moving or renaming data: it hides real changes."""
    cfg = _cfg()
    console.print(_project_line(cfg), highlight=False)
    for manifest in sorted(cfg.data.glob("**/manifest.json")):
        found = _rekey_inputs(cfg, manifest)
        if found is None:
            continue
        inputs, code = found
        m = json.loads(manifest.read_text())
        key = Stage(manifest.parent.name, inputs=inputs.copy, run=dict, root=cfg.root, code=code).key()
        new = {"inputs": key["inputs"], **({"code": key["code"]} if code else {})}
        if all(m.get(k) == v for k, v in new.items()):
            console.print(f"[dim]{cfg.rel(manifest.parent)}: up to date[/dim]")
            continue
        console.print(f"{cfg.rel(manifest.parent)}: {'would rekey' if dry_run else 'rekeyed'}")
        if not dry_run:
            manifest.write_text(json.dumps(m | new, indent=2, default=str) + "\n")


@app.command("report")
def report_cmd(project: Project = None) -> None:
    """Open the project's geometric evaluation report."""
    typer.launch(str(_cfg().stage_dir("eval") / "report.html"))


@app.command("blend")
def blend_cmd(project: Project = None) -> None:
    """Open the project's built Blender model."""
    typer.launch(str(_cfg().stage_dir("build") / "model.blend"))


if __name__ == "__main__":
    app()
