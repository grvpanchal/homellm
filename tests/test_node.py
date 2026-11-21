import importlib

import pytest


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOMELLM_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOMELLM_STATE_DIR", str(tmp_path / "state"))
    import homellm.common
    import homellm.node.agent
    importlib.reload(homellm.common)
    mod = importlib.reload(homellm.node.agent)
    homellm.common.save_yaml(mod.CONFIG_PATH, {
        "config_version": 2,
        "name": "test",
        "backends": {},
        "profiles": {"gemma": {"backend": "llamacpp", "model": "gemma.gguf"}},
    })
    return mod


def test_profiles_merge_and_delete(agent):
    node = agent.Node("tok")
    node.save_profile("qwen", {"backend": "llamacpp", "model": "q.gguf"})
    assert set(node.profiles()) == {"gemma", "qwen"}
    node.delete_profile("qwen")
    assert set(node.profiles()) == {"gemma"}
    with pytest.raises(agent.HTTPException):
        node.delete_profile("gemma")  # defined in node.yaml, not deletable from the API


def test_active_profile_persists(agent):
    node = agent.Node("tok")
    node.set_active("gemma")
    assert agent.Node("tok").active == "gemma"
    with pytest.raises(agent.HTTPException):
        node.set_active("nope")


def test_missing_backend_is_an_error_not_a_crash(agent):
    node = agent.Node("tok")
    node.set_active("gemma")
    node.tick()
    assert node.state == "error"
    assert "not enabled" in node.last_error


def test_health_endpoint_needs_no_auth_but_status_does(agent):
    from fastapi.testclient import TestClient
    app = agent.create_app("secret")
    c = TestClient(app)  # not entering the context: skips the watchdog thread and mDNS
    assert c.get("/health").json()["service"] == "homellm-node"
    assert c.get("/status").status_code == 401
    assert c.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 403
    assert c.get("/status", headers={"Authorization": "Bearer secret"}).json()["name"] == "test"


class FakeBackend:
    name = "llamacpp"

    def __init__(self, loaded=None):
        self.loaded_id = loaded
        self.stopped = False

    def available(self):
        return True

    def is_ready(self, name, profile):
        return name == self.loaded_id

    def loaded(self):
        return [self.loaded_id] if self.loaded_id else []

    def server_pid(self):
        return 4242 if self.loaded_id else None

    def unavailable_reason(self):
        return ""

    def engine_info(self):
        return {"managed": True, "tag": "b1"}

    def stop(self):
        self.stopped = True
        self.loaded_id = None

    def health(self):
        return {"warnings": []}

    def port(self):
        return 1236


def test_fresh_node_adopts_loaded_model_and_never_unloads(agent):
    node = agent.Node("tok")
    fake = FakeBackend(loaded="gemma")
    node.backends = {"llamacpp": fake}
    node.tick()
    assert node.active == "gemma" and node.state == "ready" and not fake.stopped


def test_fresh_node_with_unknown_model_stays_idle(agent):
    node = agent.Node("tok")
    fake = FakeBackend(loaded="something-else")
    node.backends = {"llamacpp": fake}
    node.tick()
    assert node.active is None and node.state == "idle" and not fake.stopped


def test_explicit_stop_unloads(agent):
    node = agent.Node("tok")
    fake = FakeBackend(loaded="gemma")
    node.backends = {"llamacpp": fake}
    node.set_active(None)
    node.tick()
    assert fake.stopped and node.state == "stopped"


def test_controller_keeps_gpu_history_and_skips_integrated(tmp_path, monkeypatch):
    from homellm.controller import app as ctl_app
    monkeypatch.setattr(ctl_app, "CONFIG_PATH", tmp_path / "controller.yaml")
    monkeypatch.setattr(ctl_app, "NODES_PATH", tmp_path / "nodes.yaml")
    ctl = ctl_app.Controller("tok", "key")
    st = {"name": "n1", "ip": "10.0.0.5", "active": "m",
          "gpus": [{"index": 0, "name": "P4", "vram_used_mb": 4000, "vram_total_mb": 7680, "util_pct": 40, "temp_c": 61},
                   {"index": 1, "name": "AMD GPU", "vram_used_mb": 200, "vram_total_mb": 512, "util_pct": 0,
                    "temp_c": None}]}
    ctl.nodes["n1"] = {"id": "n1", "name": "n1", "url": "http://10.0.0.5:8765", "online": True, "status": st}
    ctl.record_gpus("n1", st)
    ctl.record_gpus("n1", {**st, "gpus": [{**st["gpus"][0], "temp_c": 63}]})
    gpus = ctl.gpus()
    assert [g["name"] for g in gpus] == ["P4"]
    assert gpus[0]["ip"] == "10.0.0.5" and gpus[0]["history"]["temp_c"] == [61, 63]


def test_gpu_page_and_api_are_open_but_admin_api_is_not(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from homellm.controller import app as ctl_app
    monkeypatch.setattr(ctl_app, "CONFIG_PATH", tmp_path / "controller.yaml")
    monkeypatch.setattr(ctl_app, "NODES_PATH", tmp_path / "nodes.yaml")
    for k in ("HOMELLM_CLUSTER_TOKEN", "LITELLM_MASTER_KEY", "HOMELLM_ADMIN_TOKEN"):
        monkeypatch.setenv(k, "secret")
    c = TestClient(ctl_app.create_app())  # not entering the context: skips polling and mDNS
    assert c.get("/gpus").status_code == 200
    r = c.get("/api/gpus")
    assert r.status_code == 200 and r.json()["gpus"] == []
    assert c.get("/api/overview").status_code == 401
