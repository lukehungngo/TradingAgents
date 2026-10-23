"""Exercise the CLI bridge with a real local subprocess, never a paid model call."""
import json
import os
import signal
import sys
import time
from contextlib import suppress

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.llm_clients.factory import create_llm_client


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    script = tmp_path / "fake_codex.py"
    script.write_text('''import json, os, pathlib, subprocess, sys, time
args = sys.argv[1:]
if args == ["login", "status"]:
    print("Logged in using ChatGPT")
    sys.exit(0)
payload = json.loads(sys.stdin.read())
if os.environ.get("CHILD_PIDFILE"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pathlib.Path(os.environ["CHILD_PIDFILE"]).write_text(str(child.pid))
pathlib.Path(os.environ["CAPTURE"]).write_text(json.dumps({"argv":args,"payload":payload,
    "api_key_present":bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY"))}))
time.sleep(float(os.environ.get("DELAY", "0")))
if os.environ.get("FAIL"):
    print("private credential should not appear in errors", file=sys.stderr)
    sys.exit(1)
pathlib.Path(args[args.index("--output-last-message")+1]).write_text(os.environ["REPLY"])
''')
    capture = tmp_path / "capture.json"
    monkeypatch.setenv("CAPTURE", str(capture))
    monkeypatch.setenv("REPLY", json.dumps({"content": "Evidence is mixed", "tool_calls": []}))
    return [sys.executable, str(script)], capture


def client(fake_codex, **kwargs):
    command, _ = fake_codex
    return create_llm_client("codex", "default", codex_command=command, **kwargs).get_llm()


def test_factory_runs_codex_without_api_keys(fake_codex, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-use")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-use")
    result = client(fake_codex).invoke([SystemMessage(content="Analyse BTC"), HumanMessage(content="Data")])
    assert result.content == "Evidence is mixed"
    capture = json.loads(fake_codex[1].read_text())
    assert not capture["api_key_present"]
    assert capture["payload"]["messages"][0]["role"] == "system"
    assert "--ephemeral" in capture["argv"]
    assert "--ignore-user-config" in capture["argv"]
    assert "read-only" in capture["argv"]
    assert 'forced_login_method="chatgpt"' in capture["argv"]


@tool
def lookup(symbol: str) -> str:
    """Get supplied market data for a symbol."""
    return symbol


def test_tool_call_and_result_roundtrip(fake_codex, monkeypatch):
    monkeypatch.setenv("REPLY", json.dumps({"content":"", "tool_calls":[{"name":"lookup", "arguments":'{"symbol":"BTC"}'}]}))
    model = client(fake_codex).bind_tools([lookup])
    call = model.invoke("Check BTC").tool_calls[0]
    assert call["name"] == "lookup" and call["args"] == {"symbol":"BTC"}
    monkeypatch.setenv("REPLY", json.dumps({"content":"Observed", "tool_calls":[]}))
    result = model.invoke([AIMessage(content="", tool_calls=[call]), ToolMessage(content="price", tool_call_id=call["id"])])
    assert result.content == "Observed"
    messages = json.loads(fake_codex[1].read_text())["payload"]["messages"]
    assert messages[0]["tool_calls"][0]["id"] == messages[1]["tool_call_id"]


def test_structured_output_is_validated(fake_codex, monkeypatch):
    class Decision(BaseModel):
        rating: str
    monkeypatch.setenv("REPLY", json.dumps({"content":"", "tool_calls":[{"name":"Decision", "arguments":'{"rating":"HOLD"}'}]}))
    result = client(fake_codex).with_structured_output(Decision).invoke("Decide")
    assert isinstance(result, Decision) and result.rating == "HOLD"


@pytest.mark.parametrize("reply", [
    "not json",
    '{"content":"", "tool_calls":[{"name":"unknown", "arguments":"{}"}]}',
    '{"content":"", "tool_calls":[{"name":"lookup", "arguments":"[]"}]}',
])
def test_invalid_output_fails_loudly(fake_codex, monkeypatch, reply):
    monkeypatch.setenv("REPLY", reply)
    with pytest.raises(ValueError):
        client(fake_codex).bind_tools([lookup]).invoke("Check")


def test_cli_failure_does_not_echo_private_stderr(fake_codex, monkeypatch):
    monkeypatch.setenv("FAIL", "1")
    with pytest.raises(RuntimeError) as error:
        client(fake_codex).invoke("Check")
    assert "private credential" not in str(error.value)


def test_timeout_is_bounded(fake_codex, monkeypatch):
    monkeypatch.setenv("DELAY", "2")
    with pytest.raises(TimeoutError):
        client(fake_codex, codex_timeout=0.1).invoke("Check")


@pytest.mark.skipif(sys.platform != "linux", reason="Inspect Linux process state")
def test_timeout_stops_the_native_child_of_the_cli_wrapper(fake_codex, monkeypatch, tmp_path):
    from pathlib import Path

    pidfile = tmp_path / "child.pid"
    monkeypatch.setenv("CHILD_PIDFILE", str(pidfile))
    monkeypatch.setenv("DELAY", "30")
    try:
        with pytest.raises(TimeoutError):
            client(fake_codex, codex_timeout=0.5).invoke("Check")
        pid = int(pidfile.read_text())
        status = Path(f"/proc/{pid}/stat")
        for _ in range(20):
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail("Native Codex child still running after timeout")
    finally:
        if pidfile.exists():
            with suppress(ProcessLookupError):
                os.kill(int(pidfile.read_text()), signal.SIGKILL)


def test_missing_cli_explains_installation():
    with pytest.raises(RuntimeError, match="Codex CLI"):
        create_llm_client("codex", "default", codex_command=["/missing/codex"]).get_llm().invoke("Check")


def test_api_settings_rejected(fake_codex):
    with pytest.raises(ValueError):
        create_llm_client("codex", "default", base_url="https://relay.invalid", codex_command=fake_codex[0]).get_llm()


def test_required_tool_cannot_silently_become_plain_text(fake_codex):
    with pytest.raises(ValueError, match="required tool"):
        client(fake_codex).bind_tools([lookup], tool_choice="required").invoke("Check")


def test_named_tool_choice_is_enforced(fake_codex, monkeypatch):
    monkeypatch.setenv("REPLY", json.dumps({"content": "", "tool_calls": [
        {"name": "lookup", "arguments": '{"symbol":"BTC"}'},
    ]}))
    with pytest.raises(ValueError, match="tool_choice is none"):
        client(fake_codex).bind_tools([lookup], tool_choice="none").invoke("Check")


def test_tier_configuration_reaches_the_cli(fake_codex, monkeypatch):
    from tradingagents.default_config import build_default_config
    from tradingagents.llm_clients.factory import create_tier_client

    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "codex")
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "test-model")
    monkeypatch.setenv("TRADINGAGENTS_CODEX_TIMEOUT", "15")
    config = build_default_config()
    config.update(codex_command=fake_codex[0], openai_reasoning_effort="low")
    model = create_tier_client(config, "quick").get_llm()
    assert model.codex_timeout == 15
    model.invoke("Check")
    args = json.loads(fake_codex[1].read_text())["argv"]
    assert args[args.index("--model") + 1] == "test-model"
    assert 'model_reasoning_effort="low"' in args


@pytest.mark.parametrize("setting", ["api_key", "temperature", "max_tokens", "max_retries"])
def test_unsupported_settings_are_not_silently_ignored(fake_codex, setting):
    with pytest.raises(ValueError, match="Unsupported Codex CLI settings"):
        client(fake_codex, **{setting: 1})
