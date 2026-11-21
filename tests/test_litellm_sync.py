from homellm.controller.litellm_sync import deployment_id, desired_deployments, plan


def node(nid, model, ip="192.168.1.10", online=True, port=1236):
    return {"online": online, "status": {"id": nid, "name": nid, "served": {
        "model": model, "api_base": f"http://{ip}:{port}/v1", "backend": "llamacpp"}}}


def test_desired_skips_offline_and_idle():
    nodes = [node("a", "gemma"), node("b", "qwen", online=False),
             {"online": True, "status": {"id": "c", "name": "c", "served": None}}]
    d = desired_deployments(nodes)
    assert list(d) == [deployment_id("a", "gemma")]
    dep = d[deployment_id("a", "gemma")]
    assert dep["model_name"] == "gemma"
    assert dep["litellm_params"]["model"] == "openai/gemma"
    assert dep["litellm_params"]["api_base"] == "http://192.168.1.10:1236/v1"


def test_plan_adds_new_and_removes_stale_only_ours():
    desired = desired_deployments([node("a", "gemma")])
    current = [
        {"model_name": "old", "litellm_params": {"model": "openai/old", "api_base": "x"},
         "model_info": {"id": deployment_id("z", "old")}},
        {"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o"}, "model_info": {"id": "user-added"}},
    ]
    add, delete = plan(desired, current)
    assert [a["model_info"]["id"] for a in add] == [deployment_id("a", "gemma")]
    assert delete == [deployment_id("z", "old")]


def test_plan_noop_when_in_sync():
    desired = desired_deployments([node("a", "gemma")])
    current = [dict(v) for v in desired.values()]
    assert plan(desired, current) == ([], [])


def test_plan_recreates_when_node_ip_changes():
    desired = desired_deployments([node("a", "gemma", ip="192.168.1.99")])
    current = list(desired_deployments([node("a", "gemma")]).values())
    add, delete = plan(desired, current)
    dep = deployment_id("a", "gemma")
    assert delete == [dep] and add[0]["model_info"]["id"] == dep


def test_same_model_on_two_nodes_load_balances():
    d = desired_deployments([node("a", "gemma"), node("b", "gemma", ip="192.168.1.11")])
    assert len(d) == 2
    assert {v["model_name"] for v in d.values()} == {"gemma"}


def test_deployment_id_is_safe():
    assert deployment_id("abc", "org/model:q4 x") == "homellm-abc-org_model_q4_x"
