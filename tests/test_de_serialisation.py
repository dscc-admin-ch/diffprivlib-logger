import diffprivlib
from diffprivlib import models
from sklearn.pipeline import Pipeline

from diffprivlib_logger import deserialise_pipeline, serialise_pipeline


def test_serialize():
    example_pipe = Pipeline(
        [
            (
                "scaler",
                models.StandardScaler(
                    bounds=([17, 1, 0, 0, 1], [90, 160, 10000, 4356, 99])
                ),
            ),
            ("pca", models.PCA(2, data_norm=5, centered=True)),
            ("lr", models.LogisticRegression(data_norm=5)),
        ]
    )
    result_json = serialise_pipeline(example_pipe)

    expected_json = """{"module": "diffprivlib", "version": "0.6.0", "pipeline": [{"type": "_dpl_type:StandardScaler", "name": "scaler", "params": {"with_mean": true, "with_std": true, "copy": true, "epsilon": 1.0, "bounds": {"_tuple": true, "_items": [[17, 1, 0, 0, 1], [90, 160, 10000, 4356, 99]]}, "random_state": null, "accountant": "_dpl_instance:BudgetAccountant"}}, {"type": "_dpl_type:PCA", "name": "pca", "params": {"n_components": 2, "copy": true, "whiten": false, "random_state": null, "centered": true, "epsilon": 1.0, "data_norm": 5, "bounds": null, "accountant": "_dpl_instance:BudgetAccountant"}}, {"type": "_dpl_type:LogisticRegression", "name": "lr", "params": {"tol": 0.0001, "C": 1.0, "fit_intercept": true, "random_state": null, "max_iter": 100, "verbose": 0, "warm_start": false, "n_jobs": null, "epsilon": 1.0, "data_norm": 5, "accountant": "_dpl_instance:BudgetAccountant"}}]}"""  # noqa
    expected_json_updated = expected_json.replace(
        "0.6.0", diffprivlib.__version__
    )

    assert result_json == expected_json_updated


def test_serialize_deserialise():
    pipeline = Pipeline(
        [
            (
                "scaler",
                models.StandardScaler(
                    bounds=([17, 1, 0, 0, 1], [90, 160, 10000, 4356, 99])
                ),
            ),
            ("pca", models.PCA(2, data_norm=5, centered=True)),
            ("lr", models.LogisticRegression(data_norm=5)),
        ]
    )
    serialised = serialise_pipeline(pipeline)
    deserialised = deserialise_pipeline(serialised)

    for p_step, d_step in zip(pipeline.steps, deserialised.steps):
        # Same names
        assert p_step[0] == d_step[0]

        # Same values accountant
        p_step_dict = p_step[1].accountant.__dict__
        d_step_dict = d_step[1].accountant.__dict__
        assert p_step_dict == d_step_dict

        # Same other values
        p_step_dict = p_step[1].__dict__
        d_step_dict = d_step[1].__dict__
        del p_step_dict["accountant"]
        del d_step_dict["accountant"]
        assert p_step_dict == d_step_dict
