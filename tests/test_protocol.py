import numpy as np
from PIL import Image
from steel_core import KNOWN_CLASSES, UNKNOWN_CLASS, normalize_vector, make_prototype, predict_embedding

class FakeModel:
    def eval(self): return self


def test_five_known_classes_and_held_out_class_are_declared():
    assert len(KNOWN_CLASSES) == 5
    assert UNKNOWN_CLASS == "scratches"
    assert UNKNOWN_CLASS not in KNOWN_CLASSES


def test_normalize_vector_returns_unit_vector():
    v = normalize_vector([3.0, 4.0])
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_predict_flags_low_similarity_as_unknown():
    payload = {"class_names": ["crazing"], "centroids": {"crazing": [1.0, 0.0]}, "similarity_threshold": 0.9}
    result = predict_embedding(np.array([0.0, 1.0]), payload, {"classes": {}, "overrides": {}})
    assert result["label"] == "unknown"


def test_added_class_can_use_class_specific_threshold():
    payload = {"class_names": ["crazing"], "centroids": {"crazing": [1.0, 0.0]}, "similarity_threshold": 0.95}
    bank = {"classes": {"scratches": {"prototype": [0.8, 0.6], "threshold": 0.7, "status": "active"}}, "overrides": {}}
    result = predict_embedding(np.array([0.8, 0.6]), payload, bank)
    assert result["label"] == "scratches"


def test_rejected_prototype_is_not_predicted():
    payload = {"class_names": ["crazing"], "centroids": {"crazing": [1.0, 0.0]}, "similarity_threshold": 0.9}
    bank = {"classes": {"scratches": {"prototype": [0.0, 1.0], "threshold": 0.5, "status": "rejected"}}, "overrides": {}}
    result = predict_embedding(np.array([0.0, 1.0]), payload, bank)
    assert result["label"] == "unknown"
