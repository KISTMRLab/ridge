import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from example_demo import query


def test_authored_example_exposes_rule_and_fallback():
    rule=query("ridge","open both hands",{"threshold":["0.72"]})
    fallback=query("ridge","open both hands",{"threshold":["1.01"]})
    assert rule["trace"][0]["source"]=="rule"
    assert fallback["trace"][0]["source"]=="fallback"
    assert rule["slots"][0]["frames"] and "no trained weights" in rule["data_label"]
