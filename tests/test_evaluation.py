import numpy as np
import pytest
from riskshield.training.evaluation import metrics, bootstrap_intervals
from riskshield.training.factory import ClassifierFactory
from riskshield.services.prediction import PredictionService
from riskshield.domain.policy import CostPolicy


def test_calibration_and_bootstrap_are_reproducible():
    y = np.array([0, 1] * 50)
    p = np.where(y, 0.9, 0.1)
    result = metrics(y, p)
    assert result["pr_auc"] == 1
    assert result["brier"] == pytest.approx(0.01)
    assert result["ece"] == pytest.approx(0.1)
    a = bootstrap_intervals(y, p, repeats=20)
    assert a == bootstrap_intervals(y, p, repeats=20)
    assert a["pr_auc_95"] == [1.0, 1.0]
    with pytest.raises(ValueError):
        bootstrap_intervals(np.zeros(100), p)


def test_factory_preserves_all_five_weighted_models():
    models = ClassifierFactory.create(100)
    assert set(models) == {"logistic", "random_forest", "xgboost", "lightgbm", "catboost"}
    assert models["xgboost"].get_params()["scale_pos_weight"] == 100
    assert models["catboost"].get_params()["class_weights"] == [1, 100.0]


def test_batch_bounds_and_policy_validation():
    for rows in [[], [{"x": 1}] * 257, [{"x": True}], [{"x": 1e13}]]:
        with pytest.raises(ValueError):
            PredictionService.frame(rows, ["x"])
    with pytest.raises(ValueError):
        CostPolicy(review_cost=-1)
