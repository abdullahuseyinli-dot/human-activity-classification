import importlib.util
from pathlib import Path


def load_recovery_module(repository: Path):
    path = repository / "experiments" / "recover_experiment.py"
    spec = importlib.util.spec_from_file_location("recover_experiment_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_portable_pipeline_retains_all_patch_anchors(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    recovery = load_recovery_module(repository)
    notebook = recovery.load_notebook(repository / "experiments" / "pipeline_source.ipynb")
    for source_index in recovery.PIPELINE_CELLS:
        source = recovery.pipeline_cell_source(notebook, source_index)
        if source_index == 5:
            source = recovery.patch_config_cell(
                source,
                repository,
                tmp_path / "artifacts",
                repository / "data" / "manifest.csv",
            )
        if source_index == 27:
            source = recovery.patch_cv_cell(source)
        compile(source, f"pipeline-cell-{source_index}", "exec")


def test_portable_manifest_loader_guards_duplicate_split_provenance():
    repository = Path(__file__).resolve().parents[1]
    recovery = load_recovery_module(repository)
    notebook = recovery.load_notebook(repository / "experiments" / "pipeline_source.ipynb")
    source = recovery.pipeline_cell_source(notebook, 8)

    assert 'df = df.drop(columns=["original_split"])' in source
    assert "split and original_split disagree" in source
