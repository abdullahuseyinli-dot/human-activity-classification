import numpy as np
import pandas as pd

from experiments.cache_vcoco_v2_pose_oracle import pose_features


def test_pose_features_are_box_normalized_and_keep_visibility():
    row = pd.Series({"bbox_xmin": 10.0, "bbox_ymin": 20.0, "bbox_xmax": 110.0, "bbox_ymax": 220.0})
    keypoints = []
    for index in range(17):
        keypoints.extend([10.0 + index, 20.0 + 2 * index, 2])
    annotation = {"id": 1, "keypoints": keypoints}

    features = pose_features(row, annotation, [(0, 1)])

    assert features.shape == (76,)
    assert np.isclose(features[0], 0.0)
    assert np.isclose(features[1], 0.0)
    assert features[2] == 1.0
    assert features[3] == 1.0
    assert np.isclose(features[68], 0.01)
    assert np.isclose(features[69], 0.01)
