import numpy as np
import pandas as pd

from experiments.run_vcoco_v2_fewshot_curve import sample_image_groups


def test_fewshot_sampling_keeps_complete_images():
    rows = pd.DataFrame({"image_id": ["a", "a", "b", "c", "d", "d"]})
    labels = np.asarray([0, 1, 1, 2, 0, 2])

    selected, details = sample_image_groups(rows, labels, 1, np.random.default_rng(7))
    selected_groups = set(rows.iloc[selected]["image_id"])

    for group in selected_groups:
        group_indices = set(np.flatnonzero(rows["image_id"].eq(group).to_numpy()))
        assert group_indices.issubset(set(selected))
    assert details["selected_sitting"] >= 1
    assert details["selected_standing"] >= 1
    assert details["selected_walking_running"] >= 1
