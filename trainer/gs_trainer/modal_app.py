"""The Modal app: the image, the two volumes and the functions `prepare`, `depth_prior`, `pack_prior`, `train`,
`evaluate`, `gpu_test`.

Importing this module only builds descriptions (no server contact), so the CLI and the tests can import
it offline. `gs_trainer.cloud` runs these functions; nothing here is meant to be called directly.

The image is `trainer/Dockerfile` (CUDA, torch, gsplat's compiled kernels) plus, added fresh on every
run, the `gs_trainer` package and `pipeline/scanmodel/freespace.py` as the top-level module `freespace`.
Named views are uploaded only when explicitly selected for evaluation. Editing the trainer needs no
image rebuild; changing `uv.lock` or the Dockerfile does.

The trainer modules run in a subprocess (`cloud.module_cmd`): their output is streamed to the app's logs
line by line (that's what `just gs logs` and the attached `train` show), and a crash in CUDA code still
lets the function record a failed status.

A dataset's priors are made in this order, each step reading the previous one's output in `prior/`:
`priors` (mesh depth + normals) -> `calibrate` (`sigma.json`, the LiDAR's noise) in `prepare` (CPU), then
`da3` (`prior/da3/`) -> `fusion --mode all` (the fused depth and sigma of every mode into
`prior/<stem>.npz`, `fusion.json`) in `depth_prior` (GPU), which only `gs-trainer cloud push --da3` runs.
Both end by packing `prior/` into `prior.tar` (`gs_trainer.datapack`); `pack_prior` does only that, for
priors made before the tars.

`train` and `evaluate` work on a copy of the dataset on the container's disk (`_local_copy`): the volume is
slow with thousands of small files, so they read `bulk.tar` and `prior.tar` in one go and unpack them.

Every function has a timeout well above its normal run time, so nothing that hangs bills for hours: the
gsplat compile ~3 min (cap 20 min), prepare and evaluate ~10-20 min (45 min), depth_prior (45 min), the
GPU test suite (30 min); only `train` gets 3 h per attempt.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from gs_trainer import datapack
from gs_trainer.cloud import (
    APP_NAME,
    DA3_MODEL,
    DATA_MOUNT,
    DATA_VOLUME,
    GPU_FALLBACK,
    RUNS_MOUNT,
    RUNS_VOLUME,
    finish_attempt,
    gsplat_commit,
    module_cmd,
    remote_train_argv,
    start_attempt,
)

TRAINER = Path(__file__).resolve().parents[1]
REPO = TRAINER.parent
REMOTE_TESTS = "/root/tests"
COMMIT_EVERY_S = 60
LOCAL_DATA = Path("/tmp/dataset")

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)


def _compile_gsplat() -> None:
    """Image build step: JIT-compile gsplat's CUDA kernels into $TORCH_EXTENSIONS_DIR (see the Dockerfile).

    Runs from its source on a 32-core builder (Modal serialises this function, so it must stay
    self-contained). Loading the module may fail without a GPU driver; the compiled .so is what counts."""
    import os

    try:
        from gsplat.cuda.build import build_and_load_gsplat

        build_and_load_gsplat()
    except Exception as e:  # no driver on the builder: the .so is still written
        print(f"gsplat: load failed after compiling ({e}); checking the module")
    so = os.path.join(os.environ["TORCH_EXTENSIONS_DIR"], "gsplat_cuda", "gsplat_cuda.so")
    if not os.path.isfile(so):
        raise RuntimeError(f"gsplat compile failed: {so} missing")
    # At run time torch's JIT loader re-ran ninja over all 51 targets (the image snapshot doesn't keep what
    # ninja checks). gsplat first tries `from gsplat import csrc` (a prebuilt module) and only then JITs, so
    # a `csrc` shim that loads the compiled .so directly skips the build entirely.
    import gsplat

    shim = os.path.join(os.path.dirname(gsplat.__file__), "csrc.py")
    with open(shim, "w") as f:
        f.write(
            '"""The prebuilt gsplat_cuda extension (gs_trainer/modal_app.py), loaded without the JIT."""\n'
            "import importlib.util as _u\n\nimport torch as _torch  # noqa: F401  (libc10 first)\n\n"
            f"_spec = _u.spec_from_file_location('gsplat_cuda', {so!r})\n"
            "_mod = _u.module_from_spec(_spec)\n_spec.loader.exec_module(_mod)\n"
            "globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('__')})\n"
        )


# Paths are read lazily (at build/upload time, on the Mac), so this also imports inside the container,
# where they don't exist.
image = (
    modal.Image.from_dockerfile(TRAINER / "Dockerfile", context_dir=TRAINER)
    # Only the 3DGS kernels: 2DGS and 3DGUT (rolling shutter, eval3d) are the slow ones to compile (over an
    # hour on 32 cores with everything) and we don't use them. gsplat passes BUILD_* to the C++ side only,
    # hence the same macros in NVCC_FLAGS. These stay set at run time, so the load finds the build current.
    # VERBOSE=1 prints ninja's [done/total] lines into the build log.
    .env(
        {
            "BUILD_2DGS": "0",
            "BUILD_3DGUT": "0",
            "NVCC_FLAGS": "-DGSPLAT_BUILD_2DGS=0 -DGSPLAT_BUILD_3DGUT=0",
            "VERBOSE": "1",
        }
    )
    .run_function(_compile_gsplat, cpu=32, memory=64 * 1024, timeout=20 * 60)  # takes ~3 min
    # Small additions go here, after the compile, so they don't rebuild the Dockerfile's layers.
    .add_local_python_source("gs_trainer")
    .add_local_file(REPO / "pipeline" / "scanmodel" / "freespace.py", "/root/freespace.py")
)
# The test suite, for `gpu_test` only. Added at container start like `gs_trainer`, so no rebuild either.
test_image = image.add_local_dir(TRAINER / "tests", REMOTE_TESTS, ignore=["**/__pycache__/**"])


def _now() -> datetime:
    return datetime.now(UTC)


def _stream(cmd: Sequence[str]) -> int:
    """Run a command, passing its output through line by line; returns its exit code."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root"}
    p = subprocess.Popen(
        list(cmd),
        cwd="/root",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return p.wait()


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def _local_copy(src: Path, root: Path = LOCAL_DATA, *, prior: bool = True) -> Path:
    """The dataset `src` (on the volume) on the container's local disk, at `root/<hash>`.

    Reading each step's LiDAR depth and priors from the volume took ~300 ms a step (against 13 ms for the
    photometric step), and the volume is slow with thousands of small files. So `bulk.tar` (images, depth,
    confidence) and `prior.tar` (priors without DA3's raw outputs) are each copied to local disk in one
    sequential read and unpacked; the other top-level files (`transforms.json`, `split.json`,
    `points.ply`, `mesh.ply`, `views.npz`, ...) are linked. Without a tar its folders are copied file by
    file (depth, confidence, prior) or linked (images), as before the tars (`datapack.copy_plan`).
    `prior=False` leaves the priors out (the eval doesn't read them). Every step's time is printed."""
    dst = root / src.name
    ready = root / f"{src.name}.ready"  # a container reused for another call finds the copy made
    if ready.is_file():
        return dst
    if dst.exists():  # half made by an earlier call that died
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    started = time.monotonic()
    plan = datapack.copy_plan([p.name for p in src.iterdir()], prior=prior)
    for name in plan.unpack:
        t0 = time.monotonic()
        local = root / f"{src.name}-{name}"
        shutil.copyfile(src / name, local)
        t1 = time.monotonic()
        size = local.stat().st_size
        n = datapack.extract(local, dst)
        local.unlink()
        print(
            f"gs-trainer: {name} {size / 1e9:.1f} GB in {t1 - t0:.0f} s ({size / 1e6 / max(t1 - t0, 1e-3):.0f}"
            f" MB/s), {n} entries unpacked in {time.monotonic() - t1:.0f} s",
            flush=True,
        )
    for name in plan.copy:
        t0 = time.monotonic()
        shutil.copytree(src / name, dst / name, ignore=shutil.ignore_patterns(*datapack.COPY_IGNORE))
        print(
            f"gs-trainer: no tar for {name}/, copied file by file in {time.monotonic() - t0:.0f} s",
            flush=True,
        )
    for name in plan.link:
        (dst / name).symlink_to(src / name)
    if "images" in plan.link:
        print("gs-trainer: no bulk.tar, images/ is read from the volume", flush=True)
    ready.touch()
    print(f"gs-trainer: dataset ready in {dst} in {time.monotonic() - started:.0f} s", flush=True)
    return dst


def _pack_prior(data: Path) -> int:
    """Write `prior.tar` of the dataset `data` on the volume (commit afterwards); 1 if it has no priors."""
    if not (data / "prior").is_dir():
        print(f"gs-trainer: {data} has no prior/, nothing to pack", flush=True)
        return 1
    t0 = time.monotonic()
    stats = datapack.pack(data, datapack.PRIOR_TAR)
    print(
        f"gs-trainer: {datapack.PRIOR_TAR} {stats.files} files, {stats.bytes / 1e9:.2f} GB written in"
        f" {time.monotonic() - t0:.0f} s",
        flush=True,
    )
    return 0


def _drop_prior_tar(data: Path) -> None:
    """`prior/` is about to change: a stale `prior.tar` must not outlive it (the end rewrites it)."""
    (data / datapack.PRIOR_TAR).unlink(missing_ok=True)


def _commit_loop(vol: modal.Volume, stop: threading.Event, every_s: float) -> None:
    """Commit the runs volume periodically, so `ps`/`logs`/`pull` see progress and a preempted attempt's
    successor finds the last checkpoint."""
    while not stop.wait(every_s):
        try:
            vol.commit()
        except Exception as e:  # a failed commit must not end the run; the next one catches up
            print(f"gs-trainer: volume commit failed: {e}", flush=True)


@app.function(
    image=image,
    cpu=16.0,
    memory=32768,
    timeout=45 * 60,
    retries=1,
    volumes={DATA_MOUNT: data_volume},
)
def prepare(dataset_hash: str, args: list[str] | None = None) -> int:
    """Mesh priors (`gs_trainer.priors`, `args` go to it), then the LiDAR sigma table (`gs_trainer.calibrate`),
    into `datasets/<hash>/prior/`. CPU only: Open3D raycasting and numpy."""
    data = f"{DATA_MOUNT}/datasets/{dataset_hash}"
    _drop_prior_tar(Path(data))
    rc = _stream(module_cmd("gs_trainer.priors", ["--data", data, *(args or [])]))
    if rc == 0:
        rc = _stream(module_cmd("gs_trainer.calibrate", ["--data", data]))
    if rc == 0:
        rc = _pack_prior(Path(data))
    data_volume.commit()
    return rc


@app.function(
    image=image,
    gpu=GPU_FALLBACK,
    cpu=8.0,  # the fusion half is numpy on the CPU
    memory=32768,
    timeout=45 * 60,
    retries=1,  # both steps skip or redo whole files, so a retry after a preemption is safe
    volumes={DATA_MOUNT: data_volume},
)
def depth_prior(dataset_hash: str, model: str = DA3_MODEL) -> int:
    """DA3 depth for every frame (`gs_trainer.da3`, weights baked into the image under $HF_HOME), then every
    fusion mode (`gs_trainer.fusion --mode all`, which reads `prepare`'s `sigma.json`)."""
    data = f"{DATA_MOUNT}/datasets/{dataset_hash}"
    _drop_prior_tar(Path(data))
    stop = threading.Event()
    committer = threading.Thread(target=_commit_loop, args=(data_volume, stop, COMMIT_EVERY_S), daemon=True)
    committer.start()  # a preempted attempt's successor then skips the frames DA3 already did
    try:
        rc = _stream(module_cmd("gs_trainer.da3", ["--data", data, "--model", model]))
        if rc == 0:
            rc = _stream(module_cmd("gs_trainer.fusion", ["--data", data, "--mode", "all"]))
        if rc == 0:
            rc = _pack_prior(Path(data))
    finally:
        stop.set()
        committer.join()
    data_volume.commit()
    return rc


@app.function(image=image, cpu=2.0, memory=4096, timeout=45 * 60, volumes={DATA_MOUNT: data_volume})
def pack_prior(dataset_hash: str) -> int:
    """`prior.tar` for a dataset whose priors were made before the tars (`cloud push` runs it when the
    tar is missing). Reads `prior/` file by file once, so later runs don't have to."""
    rc = _pack_prior(Path(f"{DATA_MOUNT}/datasets/{dataset_hash}"))
    data_volume.commit()
    return rc


@app.function(
    image=image,
    gpu=GPU_FALLBACK,
    cpu=4.0,  # with 32 GiB about $0.45/h on top of the GPU (cloud.CPU_RAM_PER_HOUR)
    memory=32768,
    timeout=3 * 3600,  # per attempt: with retries=2 at most 9 GPU-hours if something loops
    retries=2,  # GPU functions can be preempted; `--resume auto` continues from the last checkpoint
    volumes={DATA_MOUNT: data_volume.read_only(), RUNS_MOUNT: runs_volume},
)
def train(
    dataset_hash: str, run_id: str, args: list[str], meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One training run into `/runs/<run id>/`, with `run.json` kept current for `ps`."""
    out = Path(RUNS_MOUNT) / run_id
    out.mkdir(parents=True, exist_ok=True)
    run_file = out / "run.json"
    previous = json.loads(run_file.read_text()) if run_file.is_file() else {}
    run = start_attempt(previous, meta, _now(), os.environ)
    run["call_id"] = modal.current_function_call_id()
    run["gsplat_commit"] = gsplat_commit()
    _write_json(run_file, run)
    runs_volume.commit()

    stop = threading.Event()
    committer = threading.Thread(target=_commit_loop, args=(runs_volume, stop, COMMIT_EVERY_S), daemon=True)
    committer.start()
    try:
        argv = remote_train_argv(dataset_hash, run_id, args)
        argv[argv.index("--data") + 1] = str(_local_copy(Path(argv[argv.index("--data") + 1])))
        rc = _stream(module_cmd("gs_trainer.train", argv))
    finally:
        stop.set()
        committer.join()
    run = finish_attempt(run, rc, (out / "splat.ply").is_file(), _now())
    _write_json(run_file, run)
    runs_volume.commit()
    print(f"gs-trainer: {run_id} {run['status']} (exit {rc})", flush=True)
    return {"run_id": run_id, "status": run["status"], "returncode": rc}


@app.function(
    image=image,
    gpu=GPU_FALLBACK,
    cpu=4.0,
    memory=32768,
    timeout=45 * 60,
    volumes={DATA_MOUNT: data_volume.read_only(), RUNS_MOUNT: runs_volume},
)
def evaluate(dataset_hash: str, ply_path: str, out_path: str, args: list[str] | None = None) -> int:
    """Score a PLY on the runs volume (`gs_trainer.evaluate`) into `out_path` on the runs volume."""
    runs_volume.reload()  # a PLY uploaded moments ago
    Path(out_path).mkdir(parents=True, exist_ok=True)
    data = _local_copy(Path(f"{DATA_MOUNT}/datasets/{dataset_hash}"), prior=False)
    argv = [
        "--data",
        str(data),
        "--ply",
        ply_path,
        "--out",
        out_path,
        *(args or []),
    ]
    rc = _stream(module_cmd("gs_trainer.evaluate", argv))
    runs_volume.commit()
    return rc


@app.function(image=test_image, gpu=GPU_FALLBACK, cpu=4.0, memory=32768, timeout=30 * 60)
def gpu_test(args: list[str]) -> int:
    """The trainer's test suite on a GPU (`python -m pytest -q -rs <args>` in /root, the tests in
    /root/tests), so the CUDA-only tests run too. pytest isn't in the image (it's a dev dependency):
    installed here, a few seconds, since build steps can't follow the `add_local_*` ones."""
    rc = _stream(["uv", "pip", "install", "--python", sys.executable, "pytest"])
    if rc != 0:
        return rc
    # pyproject.toml's `testpaths` isn't in the container: without it, `-k x` alone would collect /root.
    Path("/root/pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    return _stream([sys.executable, "-m", "pytest", "-q", "-rs", *args])
