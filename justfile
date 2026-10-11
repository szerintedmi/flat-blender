# Commands from the repo root. `just` lists them; arguments pass through (`just splat --train -n 7000`).
# The pipeline's own options: `just cli <command> --help`.

set positional-arguments

scanmodel := "uv run --project pipeline scanmodel"

# list the commands
default:
    @just --list --unsorted

# install the dependencies: pipeline and trainer (uv sync), viewer (bun install)
install:
    cd pipeline && uv sync
    cd trainer && uv sync
    cd viewer && bun install

# --- pipeline -------------------------------------------------------------------------------------

# any scanmodel command, e.g. `just cli ingest --project fifth --from stray ~/Downloads/capture`
cli *args:
    {{ scanmodel }} "$@"

# a Blender model from a scan: `just model fifth` (a project under data/), only what changed (--open, --glb, --poses posegraph)
model *args:
    {{ scanmodel }} model "$@"

# ingest -> poses -> refine -> fuse -> structure -> eval -> build, only what changed (--force, --raw, --skip, --poses posegraph)
all *args:
    {{ scanmodel }} all "$@"

# copy scans off the USB-connected iPad into data/<project>/raw/ (new/changed/incomplete sessions) and crash reports into logs/ipad/ (--list, -n, --force, projects...)
ipad-pull *args:
    python3 scripts/ipad_pull.py "$@"

# what is on the iPad, with sizes and status: new / downloaded / changed / incomplete (same as `just ipad-pull --list`)
ipad-list *args:
    python3 scripts/ipad_pull.py --list "$@"

# check bundles in data/<project>/raw (--project NAME, --raw <dir>, --skip <session>)
ingest *args:
    {{ scanmodel }} ingest "$@"

# splat dataset; --train trains msplat with LiDAR pruning (--trainer brush|gsplat, --preset geo, -n steps, --no-prune)
splat *args:
    {{ scanmodel }} splat "$@"

# data/<project>/scene/model.json -> data/<project>/build/model.blend (headless Blender)
build *args:
    {{ scanmodel }} build "$@"

# viewer data: model.glb, every trained splat and the named views into viewer/public/data/scans/<scan>/ (--trainer: the default splat, --viewer-default)
export *args:
    {{ scanmodel }} export "$@"

# config, tools and which stages have run
status *args:
    {{ scanmodel }} status "$@"

# open the project's geometric eval report
report *args:
    {{ scanmodel }} report "$@"

# open the project's built model in Blender
blend *args:
    {{ scanmodel }} blend "$@"

# open the project's blend, plan, report, glb or data folder: `just open plan --project fifth`
open *args:
    {{ scanmodel }} open "$@"

# regenerate the JSON Schemas after changing a model in pipeline/scanmodel/models/
schema:
    {{ scanmodel }} schema

# --- gsplat on Modal -------------------------------------------------------------------------------
# `just gs <cmd>` = `gs-trainer cloud <cmd>` (trainer/README.md, docs/plans/2026-10-08-gsplat-cloud.md):
#   setup                       check the Modal token, create the volumes, budget reminder
#   build                       build (or reuse) the image now, log shown; prints the image id
#   push [--data D]             upload the dataset under its content hash, make its priors
#   train [--name N] [--steps N] [--gpu L40S] [--max-side N] [--detach] [-- <gs_trainer.train options>]
#   sweep CONFIG...             one detached run per TOML/JSON config (`name`, `args`)
#   ps                          every run (status, step, cost estimate), the actual Modal bill
#                               (today / this month, by kind) and every live trainer app on Modal
#   logs RUN                    follow a run (Ctrl-C stops following, not the run)
#   pull RUN [--out D]          download a run (no checkpoints) with its manifest.json
#   stop RUN | --all            stop now
#   eval --ply P | --run RUN    score a splat on the shared split
#   gc [--keep 10] [--all]      delete old checkpoints; --all deletes every run and dataset
#   gpu-test [-- PYTEST ARGS]   the trainer's tests on a cloud L40S (CUDA-only ones included)
# One-time login: `uv run --project trainer modal setup` (opens a browser).

# gsplat on a Modal GPU: setup, build, push, train, sweep, ps, logs, pull, stop, eval, gc, gpu-test
gs *args:
    uv run --project trainer gs-trainer cloud "$@"

# trainer tests, types and lint (CPU, no Modal)
[working-directory: 'trainer']
gs-test:
    uv run pytest -q && uv run pyright && uv run ruff check gs_trainer tests

# --- viewer ---------------------------------------------------------------------------------------

# web viewer dev server (http://localhost:5173, or the next free port)
[working-directory: 'viewer']
viewer *args:
    bun install --silent
    bun run dev "$@"

# --- checks and benchmark -------------------------------------------------------------------------

# pipeline and trainer tests, types and lint, then the viewer's type check and build
check:
    cd pipeline && uv run pytest -q && uv run pyright && uv run ruff check
    just gs-test
    cd viewer && bun install --silent && bun run build

# ARKitScenes subset + hand-measured data/<project>/checks.json (every project's, or --project NAME; --no-arkitscenes, --rerun)
bench *args:
    cd pipeline && uv run python -m scanmodel.bench "$@"

# download the ARKitScenes bench data once (~11 GB fetched, ~3 GB kept in data/external/)
bench-download:
    cd pipeline && uv run python -m scanmodel.stages.arkitscenes download
