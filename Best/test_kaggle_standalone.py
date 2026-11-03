"""Check the uploadable notebook in isolation, without workspace helper imports."""
import ast
import copy
from pathlib import Path
import re
import runpy

import nbformat
import pytest
from IPython.core.interactiveshell import InteractiveShell

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(HERE / "solar-filament-kaggle-standalone.ipynb", as_version=4)


@pytest.fixture(scope="module")
def assembled(notebook, tmp_path_factory):
    # Only notebook code is used here. No pq50_runtime import or builder execution.
    folder = tmp_path_factory.mktemp("kaggle_notebook_only")
    shell = InteractiveShell.instance()
    shell.user_ns['get_ipython'] = lambda: shell
    register = next(c for c in notebook.cells if c.cell_type == 'code' and 'def _filament_section(' in c.source)
    result = shell.run_cell(register.source)
    assert result.error_in_exec is None and result.error_before_exec is None
    helpers = [c for c in notebook.cells if c.cell_type == 'code' and c.source.startswith('%%filament_section ')]
    for cell in helpers:
        result = shell.run_cell(cell.source)
        assert result.error_in_exec is None and result.error_before_exec is None
    script = folder / 'filament_gpu_worker.py'
    shell.user_ns['WORKER_SCRIPT'] = script
    assembly = next(c for c in notebook.cells if c.cell_type == 'code' and '_SECTION_ORDER =' in c.source)
    result = shell.run_cell(assembly.source)
    assert result.error_in_exec is None and result.error_before_exec is None
    assert script.is_file()
    # Load the generated worker in a separate module namespace, as subprocesses do.
    worker = runpy.run_path(str(script), run_name='standalone_worker_test')
    return dict(shell=shell, worker=worker, script=script, folder=folder)


def test_notebook_has_no_local_workspace_dependencies(notebook):
    nbformat.validate(notebook)
    text = '\n'.join(c.source for c in notebook.cells)
    assert 'pq50_runtime' not in text
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            source = cell.source.split('\n', 1)[1] if cell.source.startswith('%%') else cell.source
            assert not any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and
                           n.value.id == 'pipeline' for n in ast.walk(ast.parse(source)))
    assert 'results.zip' not in text
    assert '.venv' not in text
    assert not re.search(r'\b[A-Za-z]:[/\\]', text)
    assert all(not c.get('outputs') for c in notebook.cells)


def test_all_cells_are_valid_python_or_registered_magic(notebook):
    shell = InteractiveShell.instance()
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(shell.input_transformer_manager.transform_cell(cell.source), '<notebook>', 'exec')


def test_generated_worker_is_complete_and_headless(assembled):
    source = assembled['script'].read_text(encoding='utf8')
    compile(source, str(assembled['script']), 'exec')
    assert '%%filament_section' not in source and 'get_ipython' not in source
    assert '__file__' not in source
    assert source.rstrip().endswith('main()')
    for name in ['prepare', 'train_all', 'predict_part', 'calibrate', 'evaluate_report', 'export_submission']:
        assert callable(assembled['worker'][name])


def test_defaults_use_kaggle_data_and_downloadable_models(assembled):
    cfg = assembled['worker']['default_config']()
    assert cfg['mode'] == 'full' and cfg['output_root'] == '/kaggle/working'
    assert cfg['gpu_ids'] == [0, 1] and cfg['cpu_workers'] == 4
    assert cfg['pretrained'] is True and cfg['detector_weights'] == 'yolo11m.pt'
    assert [m['encoder'] for m in cfg['dense_models']] == ['resnet34', 'tu-tf_efficientnetv2_s']


def fake_input(root, name):
    folder = root / name
    (folder / 'train' / 'train_images').mkdir(parents=True)
    (folder / 'test' / 'test_images').mkdir(parents=True)
    (folder / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').write_text('{}')
    return folder


def test_data_discovery_reads_only_attached_input(assembled, tmp_path, monkeypatch):
    worker = assembled['worker']
    globals_ = worker['discover_root'].__globals__
    input_root = tmp_path / 'input'
    expected = fake_input(input_root, 'competitions/filament-segmentation-2026/MAGFilo_1.0_Kaggle_2026')
    decoy = fake_input(tmp_path, 'local_workspace')
    monkeypatch.setitem(globals_, 'INPUT_ROOT', input_root)
    monkeypatch.chdir(decoy)
    assert worker['discover_root']() == expected.resolve()
    with pytest.raises(ValueError, match='inside /kaggle/input'):
        worker['discover_root'](str(decoy))


def test_missing_or_duplicate_inputs_have_actionable_errors(assembled, tmp_path, monkeypatch):
    worker = assembled['worker']
    input_root = tmp_path / 'input'
    input_root.mkdir()
    monkeypatch.setitem(worker['discover_root'].__globals__, 'INPUT_ROOT', input_root)
    with pytest.raises(FileNotFoundError, match='Add Input'):
        worker['discover_root']()
    a, b = fake_input(input_root, 'a'), fake_input(input_root, 'b')
    with pytest.raises(ValueError, match='Multiple competition copies'):
        worker['discover_root']()
    assert worker['discover_root'](str(b)) == b.resolve()


def test_gpu_launcher_uses_automatically_created_worker(assembled, monkeypatch):
    worker = assembled['worker']
    calls = []
    monkeypatch.setitem(worker['command'].__globals__, 'WORKER_SCRIPT', assembled['script'])
    monkeypatch.setattr(worker['subprocess'], 'run', lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    config = assembled['folder'] / 'config.json'
    worker['save_json'](config, dict(gpu_ids=[0, 1]))
    worker['command'](config, 'dense', '--member', '0', distributed=True)
    cmd, kwargs = calls[0]
    assert 'torch.distributed.run' in cmd and '--nproc_per_node=2' in cmd
    assert str(assembled['script']) in cmd and kwargs['check'] is True


def test_helper_edits_are_captured_in_worker_source(assembled):
    shell = assembled['shell']
    original = shell.user_ns['_FILAMENT_SECTIONS']['models']
    updated = original + '\n_NOTEBOOK_EDIT_CAPTURE_TEST = 123\n'
    result = shell.run_cell('%%filament_section models\n' + updated)
    assert result.error_in_exec is None
    assert '_NOTEBOOK_EDIT_CAPTURE_TEST = 123' in shell.user_ns['_FILAMENT_SECTIONS']['models']
    # Keep the shared fixture's definitions/source unchanged for subsequent checks.
    shell.run_cell('%%filament_section models\n' + original)
