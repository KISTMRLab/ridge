"""Paper-method preparation hook and prepared demo endpoints on tiny synthetic BEAT fixtures."""
import argparse
import contextlib
import io
import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_fixture  # noqa: E402
import paper_method_common as pm  # noqa: E402
import prepare_paper_method as prep  # noqa: E402

SPEAKERS = ("1", "2", "3", "4")
QUICK = ["--epochs", "2", "--speakers", ",".join(SPEAKERS)]
STUB = """import json, re, sys
prompt = sys.stdin.read()
assert "Extract Key Gesture-Aligned Phrases" in prompt
words = re.findall(r'text = "([^"]+)"', prompt)
print(json.dumps({"paragraph": " ".join(words), "phrases": [" ".join(words[i:i + 4]) for i in (0, 8, 16)]}))
"""


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = prep.main(argv)
    return code, json.loads(out.getvalue().strip().splitlines()[-1])


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    base = tmp_path_factory.mktemp("ridge")
    source = beat_fixture.make_processed(base / "processed", speakers=SPEAKERS, takes=2)
    sbert = beat_fixture.make_sbert(base / "sbert")
    argv = ["--processed", str(source), "--sbert", str(sbert), "--output-root", str(base / "out"), *QUICK]
    code, result = run(argv)
    return {"code": code, "result": result, "argv": argv, "sbert": sbert, "source": source, "base": base}


def manifest_of(result):
    return json.loads((Path(result["server_args"][2]) / "manifest.json").read_text(encoding="utf-8"))


def test_target_speaker_rules_pretrain_finetune_and_heldout(prepared):
    assert prepared["code"] == 0 and prepared["result"]["ready"] is True, prepared["result"]
    manifest = manifest_of(prepared["result"])
    roles, metrics = manifest["roles"], manifest["metrics"]
    assert roles["roles"]["target"] == ["1"] and "1" not in roles["roles"]["pretrain"]
    assert not set(roles["takes"]["heldout"]) & set(roles["takes"]["target"])
    assert [s["stage"] for s in manifest["stages"]] == ["pretrain", "finetune"]
    assert metrics["strong_rules"] > 0 and metrics["annotator"] == "heuristic"
    assert metrics["finetune_pairs"] > 0 and metrics["pretrain_pairs"] > metrics["finetune_pairs"]
    assert 0 <= metrics["heldout_top1"] <= 1 and metrics["heldout_chance"] == pytest.approx(1 / metrics["heldout_pairs"], abs=1e-4)
    rules = [json.loads(x) for x in (Path(prepared["result"]["server_args"][2]) / "rules.jsonl").read_text().splitlines()]
    assert all(r["record_id"] in roles["takes"]["target"] and r["end_frame"] > r["start_frame"] for r in rules)
    assert all(len(r["embedding"]) == 32 for r in rules)


def test_cached_result_is_reused(prepared):
    code, result = run(prepared["argv"])
    assert code == 0 and result["ready"] and result["summary"]["cached"] is True


def test_raw_route_with_llm_command_and_no_pretrain(tmp_path, prepared):
    raw = beat_fixture.make_raw(tmp_path / "beat_english_v0.2.1", speakers=SPEAKERS, takes=2)
    stub = tmp_path / "llm_stub.py"; stub.write_text(STUB, encoding="utf-8")
    code, result = run(["--beat-root", str(raw), "--sbert", str(prepared["sbert"]), "--output-root", str(tmp_path / "out"),
                        "--no-pretrain", "--llm-command", f'"{sys.executable}" "{stub}"', *QUICK])
    assert code == 0 and result["ready"], result
    manifest = manifest_of(result)
    assert manifest["source"]["kind"] == "raw" and manifest["metrics"]["annotator"] == "llm"
    assert [s["stage"] for s in manifest["stages"]] == ["train"] and manifest["roles"]["roles"]["pretrain"] == []
    rows = [json.loads(x) for x in (Path(result["server_args"][2]) / "annotations.jsonl").read_text().splitlines()]
    assert rows[0]["provenance"]["prompt_sha256"] and rows[0]["phrases"]


def test_missing_source_or_sbert_is_not_ready(tmp_path, monkeypatch):
    for name in (pm.ENV_PROCESSED, pm.ENV_RAW, pm.ENV_SBERT):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(pm, "ROOT", tmp_path)
    code, result = run(["--output-root", str(tmp_path / "out")])
    assert code == 0 and result["ready"] is False and "BEAT" in result["reason"]
    monkeypatch.setattr(pm, "DEFAULT_SBERT_DIR", tmp_path / "missing")
    source = beat_fixture.make_processed(tmp_path / "p", speakers=("1", "2"), seconds=6)
    code, result = run(["--processed", str(source), "--output-root", str(tmp_path / "out")])
    assert result["ready"] is False and "save" in result["next_steps"][0].lower()


@pytest.fixture(scope="module")
def server(prepared):
    import demo_server
    args = argparse.Namespace(prepared=Path(prepared["result"]["server_args"][2]), example=False, sbert=None,
                              host="127.0.0.1", port=0)
    httpd = demo_server.make_server(args)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def test_prepared_server_rules_fallback_and_idle(server):
    library = get(server + "/api/beat-library")
    assert library["ready"] and library["prepared"] and library["clips"] and library["strong_rules"]
    rule = library["strong_rules"][0]
    for path in ("/api/beat-query", "/api/query"):
        result = get(server + path + "?" + urllib.parse.urlencode({"text": rule + " zzzz qqqq", "threshold": 0.72}))
        routes = [s["route"] for s in result["slots"]]
        assert routes[0] == "strong_rule" and "trained_text_motion_fallback" in routes
        assert result["rule_count"] == 1 and len(result["slots"][0]["frames"][0]) == 11
        assert result["metrics"]["route_counts"]["strong_rule"] == 1 and "heldout_top1" in result["metrics"]
    strict = get(server + "/api/beat-query?" + urllib.parse.urlencode({"text": rule, "threshold": 1.01, "min_similarity": 2}))
    assert strict["no_match"] is True and strict["slots"][0]["route"] == "idle_no_match"
    with pytest.raises(urllib.error.HTTPError):
        get(server + "/api/beat-query?text=")
