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


def test_normalize_image_id_handles_collated_tensor_representation():
    exporter = load_export_module()

    assert exporter.normalize_image_id("tensor(308394)") == "308394"
    assert exporter.normalize_image_id("21167") == "21167"
