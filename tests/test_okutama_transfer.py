from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.evaluate_okutama_fewshot_transfer import sampled_rows
from hac.okutama import parse_annotation, recording_metadata


def test_fewshot_sampler_is_balanced_deterministic_and_unique():
    rows = []
    for label in range(3):
        for scenario in range(4):
            for track in range(3):
                rows.append(
                    {
                        "recording_id": f"scenario-{scenario}",
                        "track_id": f"track-{label}-{scenario}-{track}",
                    }
                )
    frame = pd.DataFrame(rows)
    labels = np.repeat(np.arange(3), 12)
    indices = np.arange(len(frame))

    first = sampled_rows(frame, labels, indices, budget=7, seed=41)
    second = sampled_rows(frame, labels, indices, budget=7, seed=41)

    assert np.array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 21
    assert np.bincount(labels[first], minlength=3).tolist() == [7, 7, 7]
    assert all(
        frame.iloc[first[labels[first] == class_index]]["recording_id"].nunique() == 4
        for class_index in range(3)
    )


def test_okutama_provider_annotation_parser_preserves_actions():
    annotation = parse_annotation(
        '7 30 60 90 150 120 0 1 0 "Person" "Walking" "Carrying"'
    )

    assert annotation.track_id == 7
    assert annotation.bbox == (30, 60, 90, 150)
    assert annotation.base_actions == ("Walking",)
    assert annotation.occluded is True
    assert recording_metadata("2.1.4") == ("1.4", "drone_2", "morning")
