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
