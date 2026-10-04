import numpy as np
from ridge_gesture.pipeline import GCA,align_phrase,heuristic_phrases
from ridge_gesture.pipeline import hybrid_retrieve
def test_annotation_and_gca():
    words=[{"word":"we","start_frame":0,"end_frame":2},{"word":"move","start_frame":2,"end_frame":4},{"word":"forward","start_frame":4,"end_frame":6}]
    assert align_phrase("we move forward",words)==(0,6); assert heuristic_phrases("we move forward with strong purpose")
    text=np.array([[1.,0.],[.9,.1],[0.,1.],[.1,.9]]); motion=text.copy(); metric=GCA(2,1).fit(text,motion); assert metric.score(text,motion)>.9

def test_rule_threshold_routes_to_contrastive_fallback():
    rules=[{"phrase":"open both hands","gesture_id":"open","embedding":[1.,0.]}]
    embed=lambda _:np.array([[1.,0.]],np.float32)
    lat=np.array([[0.,1.]],np.float32)
    rule=hybrid_retrieve("open both hands",rules,embed,.9,lat,["fallback"],lambda _ : lat[0])
    fallback=hybrid_retrieve("open both hands",rules,embed,1.01,lat,["fallback"],lambda _ : lat[0])
    assert rule[0]["source"]=="rule" and rule[0]["gesture_id"]=="open"
    assert fallback[0]["source"]=="fallback" and fallback[0]["gesture_id"]=="fallback"
