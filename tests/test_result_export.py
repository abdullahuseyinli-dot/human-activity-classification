import importlib.util
from pathlib import Path


def load_export_module():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "tools" / "export_portfolio_results.py"
    spec = importlib.util.spec_from_file_location("export_portfolio_results_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_polar_export_module():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "tools" / "export_polar_final_results.py"
    spec = importlib.util.spec_from_file_location("export_polar_final_results_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_image_id_handles_collated_tensor_representation():
    exporter = load_export_module()

    assert exporter.normalize_image_id("tensor(308394)") == "308394"
    assert exporter.normalize_image_id("21167") == "21167"


def test_release_protocol_uses_portfolio_neutral_terminology():
    exporter = load_export_module()

    sanitized = exporter.sanitize_data_protocol(
        {
            "protocol": "legacy_fixed_test_plus_internal_stratified_cv",
            "final_test_rows": 43,
        }
    )

    assert sanitized == {
        "protocol": "fixed_test_plus_internal_stratified_cv",
        "final_test_rows": 43,
    }


def test_polar_export_names_remove_repeated_group_prefixes():
    exporter = load_polar_export_module()

    assert exporter.published_name("test", "test_metrics.csv") == "polar_test_metrics.csv"
    assert (
        exporter.published_name("external", "external_person_metrics.csv")
        == "polar_external_person_metrics.csv"
    )
    assert exporter.published_name("faithfulness", "summary.json") == (
        "polar_faithfulness_summary.json"
    )
