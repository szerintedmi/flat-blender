"""The dataset tars (`gs_trainer.datapack`) and the container's local copy built from them, offline."""

import json
import tarfile
from pathlib import Path

from gs_trainer import cloud, datapack
from gs_trainer.synth import write_dataset


def _with_priors(data: Path) -> Path:
    prior = data / "prior"
    (prior / "da3").mkdir(parents=True)
    (prior / "meta.json").write_text(json.dumps({"side": 640}))
    (prior / "sigma.json").write_text("{}")
    for stem in ("a", "b"):
        (prior / f"{stem}.npz").write_bytes(stem.encode() * 100)
        (prior / "da3" / f"{stem}.npz").write_bytes(b"raw")
    (prior / "c.tmp.npz").write_bytes(b"half written")
    (data / "images" / ".DS_Store").write_bytes(b"junk")
    return data


def _files(root: Path) -> dict[str, bytes]:
    return {f.relative_to(root).as_posix(): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}


def test_which_folders_go_into_which_tar(tmp_path: Path) -> None:
    data = _with_priors(write_dataset(tmp_path / "d", n_frames=4, extrap=1))
    bulk = [rel for _, rel in datapack.tar_members(data, datapack.BULK_TAR)]
    assert {r.split("/")[0] for r in bulk} == {"images", "depth", "confidence"}
    # the same files `cloud push` uploads one by one
    assert bulk == [rel for _, rel in cloud.dataset_files(data) if "/" in rel]
    prior = [rel for _, rel in datapack.tar_members(data, datapack.PRIOR_TAR)]
    assert prior == ["prior/a.npz", "prior/b.npz", "prior/meta.json", "prior/sigma.json"]  # no da3/, tmp


def test_tar_round_trip_reproduces_the_files(tmp_path: Path) -> None:
    data = _with_priors(write_dataset(tmp_path / "d", n_frames=4, extrap=1))
    out = tmp_path / "out"
    for name in (datapack.BULK_TAR, datapack.PRIOR_TAR):
        tar = tmp_path / name
        stats = datapack.pack(data, name, tar)
        members = datapack.tar_members(data, name)
        assert stats.files == len(members) and stats.bytes == sum(p.stat().st_size for p, _ in members)
        assert not tar.with_name(name + ".tmp").exists()
        with tarfile.open(tar, "r:") as t:  # uncompressed ("r:" refuses gzip & co.)
            assert len(t.getnames()) == stats.files
        assert datapack.extract(tar, out) == stats.files
    want = {
        rel: p.read_bytes()
        for name in (datapack.BULK_TAR, datapack.PRIOR_TAR)
        for p, rel in datapack.tar_members(data, name)
    }
    assert _files(out) == want


def test_pack_defaults_into_the_dataset(tmp_path: Path) -> None:
    data = _with_priors(write_dataset(tmp_path / "d", n_frames=2, extrap=0))
    datapack.pack(data, datapack.PRIOR_TAR)
    assert (data / "prior.tar").is_file()
    # a tar in the dataset never goes into the other tar or the upload
    assert all(not rel.endswith(".tar") for _, rel in datapack.tar_members(data, datapack.BULK_TAR))
    assert all(not rel.endswith(".tar") for _, rel in cloud.dataset_files(data))


def test_copy_plan_unpacks_tars_and_falls_back_per_file() -> None:
    top = ["transforms.json", "split.json", "points.ply", "mesh.ply", "views.npz", "images", "depth",
           "confidence", "prior", ".complete"]  # fmt: skip
    both = datapack.copy_plan([*top, "bulk.tar", "prior.tar"])
    assert both.unpack == ["bulk.tar", "prior.tar"]
    assert both.copy == []
    assert both.link == ["mesh.ply", "points.ply", "split.json", "transforms.json", "views.npz"]

    none = datapack.copy_plan(top)  # a dataset from before the tars: as before
    assert none.unpack == []
    assert none.copy == ["confidence", "depth", "prior"]
    assert "images" in none.link and ".complete" not in none.link

    bulk_only = datapack.copy_plan([*top, "bulk.tar", "prior.tar.tmp"])  # prior.tar being written
    assert bulk_only.unpack == ["bulk.tar"] and bulk_only.copy == ["prior"]
    assert "prior.tar.tmp" not in bulk_only.link

    prior_only = datapack.copy_plan([*top, "prior.tar"])
    assert prior_only.unpack == ["prior.tar"] and prior_only.copy == ["confidence", "depth"]

    no_prior = datapack.copy_plan([*top, "bulk.tar", "prior.tar"], prior=False)  # the eval
    assert no_prior.unpack == ["bulk.tar"] and no_prior.copy == []
    assert "prior" not in no_prior.link


def test_local_copy_from_the_tars_matches_the_per_file_copy(tmp_path: Path) -> None:
    from gs_trainer import modal_app
    from gs_trainer.data import Scene

    src = _with_priors(write_dataset(tmp_path / "vol" / "abc123", n_frames=4, extrap=1))
    plain = modal_app._local_copy(src, tmp_path / "plain")  # no tars yet: per-file copy and links
    assert plain == tmp_path / "plain" / "abc123"
    assert (plain / "images").is_symlink() and not (plain / "depth").is_symlink()

    datapack.pack(src, datapack.BULK_TAR)
    datapack.pack(src, datapack.PRIOR_TAR)
    local = modal_app._local_copy(src, tmp_path / "local")
    assert not any(p.is_symlink() for p in (local / "images", local / "depth", local / "prior"))
    assert (local / "transforms.json").is_symlink() and not (local / "bulk.tar").exists()
    assert not (local / "prior" / "da3").exists()
    expect = {k: v for k, v in _files(src).items() if not k.endswith(".tar") and "/da3/" not in k}
    expect = {k: v for k, v in expect.items() if ".tmp." not in k and ".DS_Store" not in k}
    assert _files(local) == expect
    assert len(Scene.load(local).frames) == len(Scene.load(src).frames)
    assert modal_app._local_copy(src, tmp_path / "local") == local  # made once per container

    evald = modal_app._local_copy(src, tmp_path / "eval", prior=False)
    assert not (evald / "prior").exists() and (evald / "depth").is_dir()
