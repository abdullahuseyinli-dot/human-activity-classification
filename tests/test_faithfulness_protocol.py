import importlib.util
import sys
from pathlib import Path

import pandas as pd


def load_faithfulness_module():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "experiments" / "evaluate_faithfulness.py"
    spec = importlib.util.spec_from_file_location("evaluate_faithfulness_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_oof_attribution_cohort_is_balanced_and_order_independent():
    evaluator = load_faithfulness_module()
    rows = []
    for class_index, label in enumerate(("sitting", "standing", "walking_running")):
        for index in range(15):
            rows.append(
                {
                    "image_id": f"{class_index}-{index}",
                    "label": label,
                    "fold_id": index % 5 + 1,
                }
            )
    frame = pd.DataFrame(rows)

    first = evaluator.deterministic_selection_cohort(frame, per_class=12)
    shuffled = evaluator.deterministic_selection_cohort(
        frame.sample(frac=1.0, random_state=17), per_class=12
    )

    assert first["image_id"].tolist() == shuffled["image_id"].tolist()
    assert first.groupby("label").size().to_dict() == {
        "sitting": 12,
        "standing": 12,
        "walking_running": 12,
    }


def test_raw_dino_attention_is_a_diagnostic_not_a_selection_candidate():
    evaluator = load_faithfulness_module()

    assert "attention_rollout" in evaluator.CANDIDATES["dinov2_small"]
    assert "attention_rollout" not in evaluator.ELIGIBLE["dinov2_small"]
    assert "gradient_attention_rollout" in evaluator.ELIGIBLE["dinov2_small"]
    assert evaluator.PATCH_GRID_SIZE == 16
