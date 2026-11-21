import hashlib

import httpx
import pytest

from homellm import hf

TREE = [
    {"path": "Big-Q4_K_M-00001-of-00002.gguf", "size": 10},
    {"path": "Big-Q4_K_M-00002-of-00002.gguf", "size": 5},
    {"path": "Big-UD-Q4_K_M.gguf", "size": 14},
    {"path": "Big-Q8_0.gguf", "size": 30},
    {"path": "Q6_K/Big-Q6_K-00001-of-00002.gguf", "size": 8},
    {"path": "Q6_K/Big-Q6_K-00002-of-00002.gguf", "size": 8},
    {"path": "mmproj-BF16.gguf", "size": 2},
    {"path": "mmproj-F16.gguf", "size": 2},
    {"path": "MTP/mtp-Big-Q8_0.gguf", "size": 1},
    {"path": "README.md", "size": 1},
]


def test_group_ggufs_joins_split_parts_and_skips_helpers():
    groups = {g["name"]: g for g in hf.group_ggufs(TREE)}
    assert set(groups) == {"Big-Q4_K_M", "Big-UD-Q4_K_M", "Big-Q8_0", "Big-Q6_K", "mmproj-BF16", "mmproj-F16"}
    assert groups["Big-Q4_K_M"]["paths"] == ["Big-Q4_K_M-00001-of-00002.gguf", "Big-Q4_K_M-00002-of-00002.gguf"]
    assert groups["Big-Q4_K_M"]["size"] == 15
    assert groups["Big-Q6_K"]["paths"][0].startswith("Q6_K/")
    assert groups["mmproj-F16"]["mmproj"] and not groups["Big-Q8_0"]["mmproj"]


def test_find_quant_does_not_confuse_ud_variants():
    groups = hf.group_ggufs(TREE)
    assert hf.find_quant(groups, "Q4_K_M")["name"] == "Big-Q4_K_M"
    assert hf.find_quant(groups, "ud-q4_k_m")["name"] == "Big-UD-Q4_K_M"
    with pytest.raises(hf.HFError, match="available"):
        hf.find_quant(groups, "IQ2_XXS")


def test_mmproj_prefers_f16():
    assert hf.find_mmproj(hf.group_ggufs(TREE))["name"] == "mmproj-F16"
    assert hf.find_mmproj(hf.group_ggufs(TREE), "BF16")["name"] == "mmproj-BF16"


def test_parse_hf_url():
    url = "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=1"
    assert hf.parse_hf_url(url) == ("unsloth/Qwen3-8B-GGUF", "main", "Qwen3-8B-Q4_K_M.gguf")
    assert hf.parse_hf_url("https://hf.co/a/b/blob/abc123/sub/x.gguf") == ("a/b", "abc123", "sub/x.gguf")
    assert hf.parse_hf_url("https://example.com/x.gguf") is None


def test_list_repo_rejects_bad_ids():
    with pytest.raises(hf.HFError):
        hf.list_repo("../../etc")


def test_profile_for_uses_catalog_settings(tmp_path):
    groups = hf.group_ggufs(TREE)
    plan = {"repo": "unsloth/Big-GGUF", "revision": "main", "model": hf.find_quant(groups, "Q4_K_M"),
            "mmproj": hf.find_mmproj(groups)}
    name, prof = hf.profile_for(plan, tmp_path, {"name": "big", "context": 8192, "args": ["--flash-attn", "on"]})
    assert name == "big"
    assert prof["model"] == "unsloth/Big-GGUF/Big-Q4_K_M-00001-of-00002.gguf"
    assert prof["context"] == 8192
    assert prof["args"][:2] == ["--mmproj", str(tmp_path / "unsloth" / "Big-GGUF" / "mmproj-F16.gguf")]


def test_catalog_loads_and_user_entries_override(tmp_path, monkeypatch):
    names = [e["name"] for e in hf.load_catalog()]
    assert "gemma-4-26b-a4b" in names and len(names) == len(set(names))
    user = tmp_path / "catalog.yaml"
    user.write_text("models:\n  - {name: gemma-4-26b-a4b, repo: me/x, quant: Q8_0}\n  - {name: mine, repo: me/y}\n")
    monkeypatch.setattr(hf, "USER_CATALOG", user)
    entries = {e["name"]: e for e in hf.load_catalog()}
    assert entries["gemma-4-26b-a4b"]["repo"] == "me/x" and entries["gemma-4-26b-a4b"]["mmproj"] is True
    assert "mine" in entries


def _server(data: bytes, honour_range: bool = True):
    def handler(req: httpx.Request) -> httpx.Response:
        rng = req.headers.get("range")
        if rng and honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            return httpx.Response(206, content=data[start:])
        return httpx.Response(200, content=data, headers={"content-length": str(len(data))})
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("honour_range", [True, False])
def test_download_resumes_and_verifies(tmp_path, honour_range):
    data = bytes(range(256)) * 4096
    sha = hashlib.sha256(data).hexdigest()
    dest = tmp_path / "m" / "x.gguf"
    dest.parent.mkdir()
    (dest.parent / "x.gguf.part").write_bytes(data[:1000])  # interrupted earlier
    seen = []
    hf.download_file("https://h/x", dest, len(data), sha, lambda d, t, n: seen.append(d), _server(data, honour_range))
    assert dest.read_bytes() == data
    assert not (dest.parent / "x.gguf.part").exists()
    assert seen[-1] == len(data)


def test_download_bad_hash_deletes_part(tmp_path):
    data = b"x" * 5000
    with pytest.raises(hf.HFError, match="sha256"):
        hf.download_file("https://h/x", tmp_path / "x.gguf", len(data), "0" * 64, client=_server(data))
    assert not (tmp_path / "x.gguf.part").exists() and not (tmp_path / "x.gguf").exists()


def test_download_can_be_cancelled(tmp_path):
    data = b"y" * (3 << 20)
    with pytest.raises(hf.HFError, match="cancelled"):
        hf.download_file("https://h/x", tmp_path / "x.gguf", len(data), client=_server(data),
                         should_stop=lambda: True)
    assert not (tmp_path / "x.gguf").exists()


def test_downloads_use_lmstudio_layout(tmp_path):
    assert hf.local_dir(tmp_path, "unsloth/gemma-4-26B-A4B-it-GGUF") == tmp_path / "unsloth" / "gemma-4-26B-A4B-it-GGUF"
