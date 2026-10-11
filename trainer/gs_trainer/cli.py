"""The `gs-trainer` command: the trainer modules on a local GPU, and `gs-trainer cloud ...` for Modal.

`train`, `evaluate` and `prepare` hand their arguments unchanged to the modules' own parsers (`--help`
included; `prepare` then also runs `gs_trainer.calibrate` on the same `--data`), for a machine that has a
CUDA GPU. `cloud` drives runs on Modal from the Mac; `just gs <cmd>`
is the same as `gs-trainer cloud <cmd>`. The modules are imported only when a command runs, so the CLI
starts without torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from gs_trainer import cloud

PASS_THROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}
Project = Annotated[
    str | None,
    typer.Option("--project", help="project in data/NAME [settings.toml `project`]"),
]


def local_args(args: list[str], project: str | None, *, output: str | None = None) -> list[str]:
    """Supply project paths while preserving the module's explicit options and help."""
    name = cloud.selected_project(project)
    if name is None or "--help" in args or "-h" in args:
        return args
    result = list(args)
    if not any(a == "--data" or a.startswith("--data=") for a in result):
        result += ["--data", str(cloud.project_dataset(name))]
    if output and not any(a == "--out" or a.startswith("--out=") for a in result):
        result += ["--out", str(cloud.default_run_dir(output, project=name))]
    return result


class _App(typer.Typer):
    """Turns a missing Modal token or a `CloudError` into one line instead of a traceback."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        import modal

        try:
            return super().__call__(*args, **kwargs)
        except modal.exception.AuthError as e:
            cloud.console.print("[red]No Modal token.[/red] Run `uv run --project trainer modal setup` once.")
            raise SystemExit(1) from e
        except cloud.CloudError as e:
            cloud.console.print(f"[red]{e}[/red]", markup=True, highlight=False)
            raise SystemExit(1) from e


app = _App(
    no_args_is_help=True,
    add_completion=False,
    help="The gsplat trainer: train / evaluate / prepare on a local GPU, `cloud ...` on Modal.",
)
cloud_app = typer.Typer(
    no_args_is_help=True,
    help="Runs on Modal: setup, build, push, train, sweep, ps, logs, pull, stop, eval, gc, gpu-test.",
)
app.add_typer(cloud_app, name="cloud")


@app.command(context_settings=PASS_THROUGH, add_help_option=False)
def train(ctx: typer.Context, project: Project = None) -> None:
    """Train on this machine's GPU (`python -m gs_trainer.train`; its options, see `gs-trainer train --help`)."""
    from gs_trainer import train as module

    raise typer.Exit(module.main(local_args(list(ctx.args), project, output="local")))


@app.command(context_settings=PASS_THROUGH, add_help_option=False)
def evaluate(ctx: typer.Context, project: Project = None) -> None:
    """Score a PLY on this machine's GPU (`gs_trainer.evaluate`; see `gs-trainer evaluate --help`)."""
    from gs_trainer import evaluate as module

    raise typer.Exit(module.main(local_args(list(ctx.args), project, output="evaluation")))


@app.command(context_settings=PASS_THROUGH, add_help_option=False)
def prepare(ctx: typer.Context, project: Project = None) -> None:
    """Make a dataset's mesh priors, then its LiDAR sigma table, on this machine (`gs_trainer.priors`, whose
    options these are, then `gs_trainer.calibrate`). DA3 and the fusion: `python -m gs_trainer.da3` and
    `python -m gs_trainer.fusion --mode all` (or `gs-trainer cloud push --da3`)."""
    import argparse

    from gs_trainer import calibrate, priors

    args = local_args(list(ctx.args), project)
    rc = priors.main(args)
    if rc == 0:
        data = argparse.ArgumentParser(add_help=False)
        data.add_argument("--data", required=True)
        rc = calibrate.main(["--data", data.parse_known_args(args)[0].data])
    raise typer.Exit(rc)


# --- cloud ------------------------------------------------------------------------------------------

Data = Annotated[Path | None, typer.Option("--data", help="dataset v2 dir [selected project's default]")]
Gpu = Annotated[str, typer.Option(help="Modal GPU: L40S, A100-80GB, H100, A10")]


@cloud_app.command()
def setup() -> None:
    """Check the Modal token, create the volumes, remind about the budget."""
    cloud.setup()


@cloud_app.command()
def build() -> None:
    """Build the image now (or find it cached), with the build log shown; prints the image id."""
    print(cloud.build())


@cloud_app.command()
def push(
    data: Data = None,
    project: Project = None,
    da3: Annotated[
        bool, typer.Option("--da3", help="also DA3 depth + every fusion mode (GPU), for --depth-target")
    ] = False,
    da3_model: Annotated[str, typer.Option(help="DA3 weights (Hugging Face id)")] = cloud.DA3_MODEL,
) -> None:
    """Upload the dataset under its content hash (skipped if there), then make what is missing on the
    cloud: mesh priors and the LiDAR sigma table (CPU), with --da3 also DA3 and the fused depth (GPU)."""
    print(cloud.push(data or cloud.project_dataset(project), da3, da3_model))


@cloud_app.command("train", context_settings=PASS_THROUGH)
def cloud_train(
    ctx: typer.Context,
    data: Data = None,
    project: Project = None,
    out: Annotated[Path | None, typer.Option(help="local run dir [selected project's runs/<run id>]")] = None,
    name: Annotated[str, typer.Option(help="what the run is for; part of the run id")] = "run",
    steps: Annotated[int | None, typer.Option(help="training steps [trainer default]")] = None,
    gpu: Gpu = cloud.DEFAULT_GPU,
    max_side: Annotated[int | None, typer.Option(help="training image size [trainer default]")] = None,
    detach: Annotated[
        bool, typer.Option(help="start and return; follow with `logs`, fetch with `pull`")
    ] = False,
) -> None:
    """Train on Modal: push, run, stream `step=` lines, download into --out. Options after `--` go to
    gs_trainer.train (e.g. `-- --depth-lambda 0.5`). Ctrl-C stops the run unless --detach."""
    selected = cloud.selected_project(project)
    options = {"project": selected} if selected is not None else {}
    raise typer.Exit(
        cloud.train(
            data or cloud.project_dataset(selected),
            out,
            name,
            steps,
            gpu,
            max_side,
            detach,
            list(ctx.args),
            **options,
        )
    )


@cloud_app.command()
def sweep(
    configs: Annotated[list[Path], typer.Argument(help="TOML/JSON files with `name` and `args`")],
    data: Data = None,
    project: Project = None,
    gpu: Gpu = cloud.DEFAULT_GPU,
) -> None:
    """One detached run per config, in parallel."""
    selected = cloud.selected_project(project)
    options = {"project": selected} if selected is not None else {}
    for run_id in cloud.sweep(data or cloud.project_dataset(selected), configs, gpu, **options):
        print(run_id)


@cloud_app.command()
def ps() -> None:
    """Every run (status, step, elapsed, GPU, estimated cost), the actual Modal bill (today, this month,
    by kind) and every live flat-gs app (the Modal apps keep that name)."""
    cloud.ps()


@cloud_app.command()
def logs(run: Annotated[str, typer.Argument(help="run id")]) -> None:
    """Follow a run with a progress bar. Ctrl-C stops following, not the run."""
    cloud.logs(run)


@cloud_app.command()
def pull(
    run: Annotated[str, typer.Argument(help="run id")],
    out: Annotated[Path | None, typer.Option(help="local dir [selected project's runs/<run id>]")] = None,
    project: Project = None,
    data: Annotated[Path | None, typer.Option(help="dataset dir for the manifest [the run's]")] = None,
) -> None:
    """Download a run (everything but checkpoints) and write its manifest.json."""
    try:
        selected = cloud.selected_project(project)
        options = {"project": selected} if selected is not None else {}
        cloud.pull(run, out, data, **options)
    finally:
        cloud.report_running()


@cloud_app.command()
def stop(
    run: Annotated[str | None, typer.Argument(help="run id")] = None,
    all_: Annotated[
        bool, typer.Option("--all", help="stop every flat-gs* app on Modal (runs, evals, builds)")
    ] = False,
) -> None:
    """Stop a run (or all of them) now."""
    if not run and not all_:
        raise typer.BadParameter("give a run id or --all")
    cloud.stop(run, all_)


@cloud_app.command("eval", context_settings=PASS_THROUGH)
def cloud_eval(
    ctx: typer.Context,
    project: Project = None,
    data: Annotated[
        Path | None,
        typer.Option(
            "--data",
            help="dataset v2 dir (`scanmodel splat`) [a run's own dataset, else the selected project's default]",
        ),
    ] = None,
    ply: Annotated[Path | None, typer.Option(help="a local PLY (msplat, Brush, ...)")] = None,
    run: Annotated[str | None, typer.Option(help="a cloud run id")] = None,
    out: Annotated[Path | None, typer.Option(help="local dir for metrics and renders")] = None,
) -> None:
    """Score a splat on the shared split on a cloud GPU. Options after `--` go to gs_trainer.evaluate."""
    if (ply is None) == (run is None):
        raise typer.BadParameter("give exactly one of --ply or --run")
    selected = cloud.selected_project(project)
    options = {"project": selected} if selected is not None else {}
    cloud.evaluate(
        data if data or run else cloud.project_dataset(selected), ply, run, out, list(ctx.args), **options
    )


@cloud_app.command()
def gc(
    keep: Annotated[int, typer.Option(help="runs whose checkpoints are kept, newest first")] = 10,
    all_: Annotated[bool, typer.Option("--all", help="delete both volumes: every run and dataset")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="don't ask before --all")] = False,
) -> None:
    """Delete old runs' checkpoints; with --all, every run and dataset."""
    cloud.gc(keep, all_, yes)


@cloud_app.command("gpu-test", context_settings=PASS_THROUGH)
def gpu_test(ctx: typer.Context) -> None:
    """Run the trainer's tests on a cloud L40S (CUDA-only tests included). Options after `--` go to
    pytest (e.g. `-- -k render -x`). Exit code: pytest's."""
    raise typer.Exit(cloud.gpu_test(list(ctx.args)))


if __name__ == "__main__":
    app()
