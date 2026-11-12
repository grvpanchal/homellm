import io
import tarfile

import pytest

from homellm import engine


def release(tag="b100"):
    names = [f"llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz",
             f"llama-{tag}-bin-ubuntu-cuda-13.5-x64.tar.gz", f"cudart-llama-{tag}-bin-ubuntu-cuda-13.5-x64.tar.gz",
             f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", f"llama-{tag}-bin-ubuntu-x64.tar.gz",
             f"llama-{tag}-bin-ubuntu-rocm-10.0-x64.tar.gz", f"llama-{tag}-bin-ubuntu-arm64.tar.gz",
             f"llama-{tag}-bin-ubuntu-openvino-2026.4.1-x64.tar.gz", f"llama-{tag}-bin-win-cuda-12.4-x64.zip"]
    return {"tag_name": tag, "assets": [{"name": n, "browser_download_url": f"https://x/{n}"} for n in names]}


@pytest.mark.parametrize("flavor,expect", [
    ("cuda-12", ["llama-b100-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b100-bin-ubuntu-cuda-12.8-x64.tar.gz"]),
    ("cuda-13", ["llama-b100-bin-ubuntu-cuda-13.5-x64.tar.gz", "cudart-llama-b100-bin-ubuntu-cuda-13.5-x64.tar.gz"]),
    ("vulkan", ["llama-b100-bin-ubuntu-vulkan-x64.tar.gz"]),
    ("rocm", ["llama-b100-bin-ubuntu-rocm-10.0-x64.tar.gz"]),
    ("cpu", ["llama-b100-bin-ubuntu-x64.tar.gz"]),
])
def test_pick_assets(flavor, expect):
    assert [a["name"] for a in engine.pick_assets(release(), flavor, "x64")] == expect


def test_pick_assets_needs_cuda_runtime():
    rel = release()
    rel["assets"] = [a for a in rel["assets"] if not a["name"].startswith("cudart-")]
    assert engine.pick_assets(rel, "cuda-12", "x64") is None


def test_pascal_gets_cuda_12(monkeypatch):
    monkeypatch.setattr(engine, "_arch", lambda: "x64")
    monkeypatch.setattr(engine, "nvidia_compute_caps", lambda: [6.1])
    assert engine.detect_flavor() == "cuda-12"
    monkeypatch.setattr(engine, "nvidia_compute_caps", lambda: [8.6, 6.0])
    assert engine.detect_flavor() == "cuda-12"  # mixed: the oldest card decides
    monkeypatch.setattr(engine, "nvidia_compute_caps", lambda: [8.9])
    assert engine.detect_flavor() == "cuda-13"
    monkeypatch.setattr(engine, "nvidia_compute_caps", lambda: [])
    monkeypatch.setattr(engine, "amd_gpu", lambda: False)
    monkeypatch.setattr(engine, "has_vulkan", lambda: False)
    assert engine.detect_flavor() == "cpu"


def make_tar(path, members):
    with tarfile.open(path, "w:gz") as t:
        for name, kind, payload in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                info.mode = 0o755
                t.addfile(info, io.BytesIO(payload))
            elif kind == "sym":
                info.type = tarfile.SYMTYPE
                info.linkname = payload
                t.addfile(info)
            else:
                info.type = tarfile.DIRTYPE
                t.addfile(info)


def test_safe_extract_flattens_top_folder(tmp_path):
    a = tmp_path / "a.tar.gz"
    make_tar(a, [("llama-b1/", "dir", None), ("llama-b1/llama-server", "file", b"bin"),
                 ("llama-b1/libggml.so.0.1", "file", b"lib"), ("llama-b1/libggml.so", "sym", "libggml.so.0.1")])
    out = tmp_path / "out"
    engine.safe_extract(a, out)
    assert (out / "llama-server").read_bytes() == b"bin"
    assert (out / "libggml.so").is_symlink() and (out / "libggml.so").read_bytes() == b"lib"


@pytest.mark.parametrize("members", [
    [("top/../../evil", "file", b"x")],
    [("top/link", "sym", "/etc/passwd")],
    [("top/link", "sym", "../../outside")],
])
def test_safe_extract_refuses_escapes(tmp_path, members):
    a = tmp_path / "a.tar.gz"
    make_tar(a, members)
    with pytest.raises(engine.EngineError):
        engine.safe_extract(a, tmp_path / "out")
    assert not (tmp_path / "evil").exists()


def test_install_switches_current_only_after_check(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "ENGINE_DIR", tmp_path)
    monkeypatch.setattr(engine, "CURRENT", tmp_path / "current")
    rel = {"tag_name": "b7", "assets": [{"name": "llama-b7-bin-ubuntu-x64.tar.gz", "size": 0,
                                         "browser_download_url": "https://x/a"}]}
    monkeypatch.setattr(engine, "find_release", lambda flavor, tag, c: (rel, rel["assets"]))

    def fake_download(url, dest, size, sha, progress, client):
        dest.parent.mkdir(parents=True, exist_ok=True)
        make_tar(dest, [("llama-b7/llama-server", "file", b"#!/bin/sh\necho 'version: 7 (abc)'\n")])
        return dest

    monkeypatch.setattr(engine, "download_file", fake_download)
    cur = engine.install("cpu", log=lambda m: None)
    assert cur["tag"] == "b7" and cur["flavor"] == "cpu" and "version: 7" in cur["version"]
    assert (tmp_path / "current").resolve() == (tmp_path / "b7-cpu").resolve()

    # a build that cannot run leaves `current` where it was
    rel["tag_name"] = "b8"
    rel["assets"][0]["name"] = "llama-b8-bin-ubuntu-x64.tar.gz"

    def broken(url, dest, size, sha, progress, client):
        dest.parent.mkdir(parents=True, exist_ok=True)
        make_tar(dest, [("llama-b8/llama-server", "file", b"#!/bin/sh\nexit 1\n")])
        return dest

    monkeypatch.setattr(engine, "download_file", broken)
    with pytest.raises(engine.EngineError):
        engine.install("cpu", log=lambda m: None)
    assert engine.current()["tag"] == "b7"
    assert not (tmp_path / "b8-cpu").exists()
