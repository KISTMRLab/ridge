import numpy as np
from ridge_gesture.pipeline import GCA,align_phrase,heuristic_phrases
def test_annotation_and_gca():
    words=[{"word":"we","start_frame":0,"end_frame":2},{"word":"move","start_frame":2,"end_frame":4},{"word":"forward","start_frame":4,"end_frame":6}]
    assert align_phrase("we move forward",words)==(0,6); assert heuristic_phrases("we move forward with strong purpose")
    text=np.array([[1.,0.],[.9,.1],[0.,1.],[.1,.9]]); motion=text.copy(); metric=GCA(2,1).fit(text,motion); assert metric.score(text,motion)>.9

