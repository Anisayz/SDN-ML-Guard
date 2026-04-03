# tests/test_predict.py
"""
Unit tests for predict.py
Run with: pytest tests/test_predict.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import get_engine, Verdict

@pytest.fixture(scope="module")
def engine():
    """Load engine once for all tests in this module."""
    return get_engine()

def test_engine_loads(engine):
    """Engine loads all artefacts without error."""
    assert engine._loaded
    assert engine.clf is not None
    assert engine.scaler is not None
    assert engine.le is not None
    assert len(engine.feature_list) == 72

def test_predict_returns_verdict(engine):
    """predict() returns a Verdict for a zero flow."""
    flow = {f: 0.0 for f in engine.feature_list}
    v = engine.predict(flow)
    assert isinstance(v, Verdict)
    assert v.verdict in ("ATTACK", "BENIGN", "ANOMALY", "SUSPECT")
    assert 0.0 <= v.confidence <= 1.0
    assert v.label in engine.le.classes_

def test_predict_missing_features(engine):
    """predict() handles missing features gracefully — fills with 0."""
    v = engine.predict({})   # completely empty dict
    assert isinstance(v, Verdict)
    assert v.verdict in ("ATTACK", "BENIGN", "ANOMALY", "SUSPECT")

def test_predict_extra_features(engine):
    """predict() ignores unknown feature keys."""
    flow = {f: 1.0 for f in engine.feature_list}
    flow["nonexistent_feature"] = 999.0
    v = engine.predict(flow)
    assert isinstance(v, Verdict)

def test_predict_batch_length(engine):
    """predict_batch() returns same number of verdicts as input flows."""
    flows = [{f: 0.0 for f in engine.feature_list} for _ in range(10)]
    verdicts = engine.predict_batch(flows)
    assert len(verdicts) == 10

def test_predict_batch_all_verdicts(engine):
    """All batch results are valid Verdict objects."""
    flows = [{f: float(i) for f in engine.feature_list} for i in range(5)]
    for v in engine.predict_batch(flows):
        assert v.verdict in ("ATTACK", "BENIGN", "ANOMALY", "SUSPECT")
        assert v.label in engine.le.classes_

def test_verdict_to_dict(engine):
    """to_dict() returns all expected keys."""
    flow = {f: 0.0 for f in engine.feature_list}
    v = engine.predict(flow)
    d = v.to_dict()
    assert "label" in d
    assert "confidence" in d
    assert "verdict" in d
    assert "is_attack" in d
    assert "class_probabilities" in d

def test_nan_inf_handling(engine):
    """NaN and Inf in input features don't crash predict()."""
    import math
    flow = {f: float("nan") for f in engine.feature_list}
    flow[engine.feature_list[0]] = float("inf")
    v = engine.predict(flow)
    assert isinstance(v, Verdict)

def test_singleton(engine):
    """get_engine() always returns the same instance."""
    engine2 = get_engine()
    assert engine is engine2