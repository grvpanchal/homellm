from homellm import initcfg
from homellm.common import load_yaml, save_yaml


def setup(tmp_path, profiles, extra=None, active=None):
    cfg_dir, state_dir, lm = tmp_path / "cfg", tmp_path / "state", tmp_path / "lmstudio"
    models = tmp_path / "models"
    models.mkdir()
    (models / "own.gguf").write_bytes(b"x")
    d = lm / "lmstudio-community" / "gemma-4-26B-A4B-it-GGUF"
    d.mkdir(parents=True)
    (d / "gemma-4-26B-A4B-it-Q4_K_M.gguf").write_bytes(b"x" * 10)
    (d / "mmproj-gemma-4-26B-A4B-it-BF16.gguf").write_bytes(b"x")
    save_yaml(cfg_dir / "node.yaml", {
        "name": "n",
        "backends": {"lmstudio": {"enabled": True, "lms": "~/.lmstudio/bin/lms"},
                     "llamacpp": {"enabled": True, "server_bin": "/opt/llama-server", "models_dir": str(models)}},
        "profiles": profiles})
    if extra is not None:
        save_yaml(cfg_dir / "profiles.yaml", extra)
    save_yaml(state_dir / "node-state.yaml", {"active": active, "stopped": False})
    return cfg_dir, state_dir, lm, d


def test_lmstudio_profile_becomes_llamacpp_on_same_file(tmp_path, monkeypatch):
    monkeypatch.setattr(initcfg.engine, "current", lambda: None)
    cfg_dir, state_dir, lm, d = setup(tmp_path, {
        "gemma-4-26b-a4b": {"backend": "lmstudio", "model": "google/gemma-4-26b-a4b", "context": 50176, "parallel": 2},
        "own": {"backend": "llamacpp", "model": "own.gguf"}}, active="gemma-4-26b-a4b")
    notes = initcfg.migrate_node(cfg_dir, state_dir, lm)
    cfg = load_yaml(cfg_dir / "node.yaml")
    assert "lmstudio" not in cfg["backends"] and cfg["config_version"] == 2
    p = cfg["profiles"]["gemma-4-26b-a4b"]
    assert p["backend"] == "llamacpp" and p["model"] == str(d / "gemma-4-26B-A4B-it-Q4_K_M.gguf")
    assert p["context"] == 50176 and p["parallel"] == 2 and "--mmproj" in p["args"]
    assert cfg["backends"]["llamacpp"]["extra_models_dirs"] == [str(lm)]
    # engine not installed yet: keep the working custom binary
    assert cfg["backends"]["llamacpp"]["server_bin"] == "/opt/llama-server"
    assert load_yaml(state_dir / "node-state.yaml")["active"] == "gemma-4-26b-a4b"
    assert (cfg_dir / "node.yaml.bak-v1").exists() and notes
    assert initcfg.migrate_node(cfg_dir, state_dir, lm) == []  # runs once


def test_existing_llamacpp_twin_takes_the_lmstudio_name(tmp_path, monkeypatch):
    monkeypatch.setattr(initcfg.engine, "current", lambda: {"dir": "/x", "server_bin": "/x/llama-server"})
    lm_file = tmp_path / "lmstudio/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf"
    cfg_dir, state_dir, lm, d = setup(tmp_path, {
        "gemma-4-26b-a4b": {"backend": "lmstudio", "model": "google/gemma-4-26b-a4b"},
        "gemma-4-26b-a4b-llamacpp": {"backend": "llamacpp", "model": str(lm_file), "args": ["--kv-unified"]}},
        active="gemma-4-26b-a4b-llamacpp")
    initcfg.migrate_node(cfg_dir, state_dir, lm)
    cfg = load_yaml(cfg_dir / "node.yaml")
    assert set(cfg["profiles"]) == {"gemma-4-26b-a4b"}
    assert cfg["profiles"]["gemma-4-26b-a4b"]["args"] == ["--kv-unified"]
    assert load_yaml(state_dir / "node-state.yaml")["active"] == "gemma-4-26b-a4b"
    assert "server_bin" not in cfg["backends"]["llamacpp"]  # switched to homellm's own engine


def test_missing_lmstudio_model_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(initcfg.engine, "current", lambda: None)
    cfg_dir, state_dir, lm, _ = setup(tmp_path, {"own": {"backend": "llamacpp", "model": "own.gguf"}},
                                      extra={"qwen": {"backend": "lmstudio", "model": "qwen/qwen3-14b"}},
                                      active="qwen")
    notes = initcfg.migrate_node(cfg_dir, state_dir, lm)
    assert load_yaml(cfg_dir / "profiles.yaml") == {}
    assert load_yaml(state_dir / "node-state.yaml")["active"] is None
    assert any("not found" in n for n in notes)


def test_missing_models_dir_falls_back_to_lmstudio(tmp_path):
    from homellm.common import load_yaml, save_yaml
    from homellm.initcfg import migrate_node
    cfg_dir, lm = tmp_path / "cfg", tmp_path / "lmstudio"
    lm.mkdir()
    save_yaml(cfg_dir / "node.yaml", {"config_version": 2, "backends": {"llamacpp": {
        "models_dir": str(tmp_path / "models"), "extra_models_dirs": [str(lm)]}}, "profiles": {}})
    notes = migrate_node(cfg_dir, tmp_path / "state", lm_root=lm)
    lc = load_yaml(cfg_dir / "node.yaml")["backends"]["llamacpp"]
    assert lc["models_dir"] == str(lm) and "extra_models_dirs" not in lc and notes
    assert migrate_node(cfg_dir, tmp_path / "state", lm_root=lm) == []  # idempotent
