"""Integration checks for the 0-33 notebook's cached dual-T4 input pipeline.

Use Ultralytics 8.4.171. FILAMENT_ULTRALYTICS_PATH may point to an isolated install.
No competition training, pretrained weight download, or submission upload is run.
"""
import ast
import copy
import hashlib
import importlib
import json
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get('FILAMENT_ULTRALYTICS_PATH'):
    sys.path.insert(0, os.environ['FILAMENT_ULTRALYTICS_PATH'])

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'Final/solar-filament-0-33.ipynb'
REFERENCE = ROOT / 'Final/solar-filament-pre-processing.ipynb'


@pytest.fixture(scope='module')
def ns(tmp_path_factory):
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    space = {'__name__': 'training_notebook_test', 'SUBMISSION_PATH': 'unused_test_submission.csv'}
    for index in [6, 8]:
        exec(''.join(notebook['cells'][index]['source']), space)
    runtime_root = tmp_path_factory.mktemp('solar_runtime')
    name = 'solar_test_runtime_' + hashlib.sha256(space['PREPROCESSING_RUNTIME_SOURCE'].encode()).hexdigest()[:16]
    (runtime_root / (name + '.py')).write_text(space['PREPROCESSING_RUNTIME_SOURCE'], encoding='utf-8')
    sys.path.insert(0, str(runtime_root))
    space['solar_runtime'] = importlib.import_module(name)
    space['StreamingSolarSegmentationTrainer'] = space['solar_runtime'].StreamingSolarSegmentationTrainer
    for index in [9, 11, 13, 14, 21, 25, 27, 31]:
        tree = ast.parse(''.join(notebook['cells'][index]['source']))
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, f'{NOTEBOOK.name}:{index}', 'exec'), space)
    exec(''.join(notebook['cells'][20]['source']), space)
    return space


@pytest.fixture
def cfg(ns):
    result = copy.deepcopy(ns['PREPROCESSING_CONFIG'])
    result.update(expected_shape=[256, 256], hough_coarse_size=256, hough_fine_size=256,
                  radial_bins=32, radial_sample_stride=2, polynomial_sample_stride=4)
    return result


def synthetic_sun(size=256):
    yy, xx = np.ogrid[:size, :size]
    radius = np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2) / (.43 * size)
    image = np.where(radius <= 1, 180 - 55 * radius**2 + 8 * xx / size, 0)
    image[(np.abs(yy - .51 * size) < max(2, size // 100)) & (xx > .30 * size) & (xx < .70 * size)] -= 65
    return np.clip(image, 0, 255).astype(np.uint8)


def test_notebook_valid_and_reference_functions_preserved(ns):
    import nbformat
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is None and cell.outputs == []
            if not cell.source.lstrip().startswith('%pip'):
                ast.parse(cell.source)
    original_functions = {}
    original_config = None
    for cell in nbformat.read(REFERENCE, as_version=4).cells:
        if cell.cell_type != 'code':
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.FunctionDef):
                original_functions[node.name] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in node.targets):
                original_config = ast.literal_eval(node.value)
    embedded = ast.parse(ns['PREPROCESSING_RUNTIME_SOURCE'])
    for node in embedded.body:
        if isinstance(node, ast.FunctionDef) and node.name in original_functions:
            assert ast.dump(node, include_attributes=False) == original_functions[node.name]
    assert original_config is not None
    for key, value in original_config.items():
        if key not in {'save_intermediates', 'save_float_stages'}:
            assert ns['PREPROCESSING_CONFIG'][key] == value
    assert ns['TRAIN_OPTIONS']['hsv_v'] == ns['TRAIN_OPTIONS']['bgr'] == 0
    assert ns['TRAIN_OPTIONS']['augmentations'] == []
    assert ns['PREDICTION_CACHE_POLICY'] == 'stream'
    assert ns['CLEANUP_INTERMEDIATES']
    assert ns['NUM_EPOCHS'] == 100  # preserve the user's configured training budget
    assert ns['REQUIRE_TWO_T4'] and ns['PRECOMPUTE_INPUTS']
    assert ns['CUDA_BATCH_PREFETCH'] and ns['TRAIN_PREFETCH_FACTOR'] == 4
    assert not ns['TRAIN_OPTIONS']['deterministic']


@pytest.mark.parametrize('mode', ['multi_channel', 'enhanced', 'original'])
def test_input_modes_have_exact_reference_values_and_bgr_storage(ns, cfg, mode):
    image = synthetic_sun()
    unchanged = image.copy()
    stages = ns['solar_runtime'].preprocess_image(image, cfg)
    bgr = ns['solar_runtime'].preprocess_model_input(image, cfg, mode)
    rgb = bgr[..., ::-1]
    if mode == 'multi_channel':
        assert np.array_equal(rgb, stages['multi_channel'])
    else:
        gray = stages['enhanced'] if mode == 'enhanced' else image
        assert np.array_equal(rgb, np.repeat(gray[..., None], 3, axis=-1))
    assert bgr.dtype == np.uint8 and bgr.flags.c_contiguous
    assert np.array_equal(image, unchanged)


def make_workspace(ns, tmp_path, cfg, count=2):
    source = tmp_path / 'originals'
    workspace = tmp_path / 'experiment/yolo_filament_dataset'
    source.mkdir(parents=True)
    for split in ['train', 'val']:
        (workspace / 'images' / split).mkdir(parents=True)
        (workspace / 'labels' / split).mkdir(parents=True)
        for i in range(count):
            name = f'{split}_{i}.png'
            path = source / name
            ns['cv2'].imwrite(str(path), synthetic_sun())
            ns['link_source_image'](path, workspace / 'images' / split / name)
            image = ns['solar_runtime'].load_grayscale(path, cfg)
            processed = ns['solar_runtime'].preprocess_model_input(image, cfg, 'multi_channel')
            prepared = workspace / 'input_cache' / split / (Path(name).stem + '.npy')
            prepared.parent.mkdir(parents=True, exist_ok=True)
            np.save(prepared, processed, allow_pickle=False)
            (workspace / 'labels' / split / (Path(name).stem + '.txt')).write_text(
                '0 .30 .47 .70 .47 .70 .55 .30 .55\n', encoding='utf-8')
    ns['save_json'](workspace / 'streaming_config.json', dict(
        preprocessing=cfg, input_mode='multi_channel', run_fingerprint='test',
        prepared_inputs=True, prefetch_factor=4, cuda_prefetch=True,
        validation_workers=0, validation_batch_size=1))
    yaml = workspace / 'filament.yaml'
    yaml.write_text(f'path: {json.dumps(workspace.as_posix())}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n')
    return source, workspace, yaml


def test_real_dataset_loads_before_resize_and_ignores_all_image_caches(ns, cfg, tmp_path):
    from ultralytics.cfg import get_cfg
    _, workspace, yaml = make_workspace(ns, tmp_path, cfg)
    args = get_cfg(overrides=dict(data=str(yaml), task='segment', imgsz=128,
                                  mosaic=0., mixup=0., copy_paste=0.,
                                  hsv_h=0., hsv_s=0., hsv_v=0., bgr=0., augmentations=[]))
    dataset = ns['solar_runtime'].build_streaming_dataset(
        args, str(workspace / 'images/train'), 2, {'names': {0: 'filament'}, 'nc': 1, 'channels': 3}, 'train', 32)
    original = ns['solar_runtime'].load_grayscale(dataset.im_files[0], cfg)
    expected = ns['solar_runtime'].preprocess_model_input(original, cfg, 'multi_channel')
    expected = ns['cv2'].resize(expected, (128, 128), interpolation=ns['cv2'].INTER_LINEAR)
    # Neither a stale RAM entry nor an adjacent NPY image may bypass the pipeline.
    dataset.ims[0] = np.full((128, 128, 3), 255, np.uint8)
    dataset.npy_files[0].write_bytes(b'invalid legacy npy cache')
    image, shape, resized = dataset.load_image(0)
    dataset.ims[0] = None
    assert np.array_equal(image, expected)
    assert shape == (256, 256) and resized == (128, 128)
    dataset.npy_files[0].unlink()
    for i in range(len(dataset)):
        dataset.load_image(i)
    assert not dataset.cache and not dataset.buffer
    assert all(entry is None for entry in dataset.ims)
    formatted = dataset[0]
    assert formatted['img'].shape == (3, 128, 128)
    assert formatted['masks'].ndim == 3 and formatted['masks'].numel() > 0


def test_cached_loader_never_runs_filters_and_does_not_mutate_cache(ns, cfg, tmp_path, monkeypatch):
    from ultralytics.cfg import get_cfg
    _, workspace, yaml = make_workspace(ns, tmp_path, cfg)
    args = get_cfg(overrides=dict(data=str(yaml), task='segment', imgsz=256,
                                  mosaic=0., mixup=0., copy_paste=0., augmentations=[]))
    runtime = ns['solar_runtime']
    dataset = runtime.build_streaming_dataset(args, str(workspace / 'images/train'), 2,
                                              {'names': {0: 'filament'}, 'nc': 1, 'channels': 3}, 'train', 32)
    path = workspace / 'input_cache/train/train_0.npy'
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU preprocessing entered the cached hot path')
    monkeypatch.setattr(runtime, 'preprocess_model_input', forbidden)
    monkeypatch.setattr(runtime, 'load_grayscale', forbidden)
    first = dataset.load_image(0)[0]
    assert first.flags.writeable
    first[:] = 255  # geometric/color array operations cannot change the disk cache
    second = dataset.load_image(0)[0]
    assert not np.all(second == 255)
    assert path.read_bytes() == before
    path.unlink()
    with pytest.raises(FileNotFoundError, match='Prepared input missing'):
        dataset.load_image(0)


def test_preparation_cache_reuse_auxiliary_values_and_parameter_invalidation(ns, cfg, tmp_path, monkeypatch):
    source = tmp_path / 'sun.png'
    ns['cv2'].imwrite(str(source), synthetic_sun())
    monkeypatch.setitem(ns, 'YOLO_DATA_DIR', tmp_path / 'generated')
    monkeypatch.setitem(ns, 'PREPROCESSING_CONFIG', cfg)
    calls = []
    actual = ns['preprocess_model_input']
    def counted(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)
    monkeypatch.setitem(ns, 'preprocess_model_input', counted)
    first = ns['cache_prepared_input'](source, 'train')
    second = ns['cache_prepared_input'](source, 'train')
    assert first == second and len(calls) == 1
    bgr, aux = ns['model_input_for_path'](source, return_aux=True)
    assert len(calls) == 1
    assert np.array_equal(bgr[..., 0], aux['candidates'])
    assert aux['disk'] == first['disk']
    cfg['clahe_clip_limit'] = 3.0
    changed = ns['cache_prepared_input'](source, 'train')
    assert changed['signature'] != first['signature'] and len(calls) == 2


def test_target_guard_stops_local_tuning_before_loading_a_model(ns, monkeypatch):
    monkeypatch.setitem(ns, 'GPU_IDS', [])
    monkeypatch.setitem(ns, 'REQUIRE_TWO_T4', True)
    def forbidden(*args, **kwargs):
        raise AssertionError('No model probe should run on local hardware')
    monkeypatch.setitem(ns, 'YOLO', forbidden)
    with pytest.raises(RuntimeError, match='Kaggle accelerator GPU T4 x2'):
        ns['choose_dual_t4_batch']()


@pytest.mark.parametrize('estimate, warning_kind, expected_per_gpu', [
    (3, None, 3),
    (4, None, 4),
    (3, 'rank', 1),
    (4, 'rank', 1),
    (16, None, 1),  # Ultralytics' internal fallback is not a validated batch of four.
    (0, None, 1),
    (-1, None, 1),
    (float('inf'), None, 1),
    (float('nan'), None, 1),
    (3, 'other', 3),
])
def test_dual_t4_batch_probe_uses_weaker_gpu_and_returns_ddp_divisible_batch(
        ns, monkeypatch, estimate, warning_kind, expected_per_gpu):
    from ultralytics.utils import autobatch
    events = {}
    class FakeNetwork:
        def to(self, device):
            events['device'] = str(device)
            return self
    monkeypatch.setitem(ns, 'GPU_IDS', [0, 1])
    monkeypatch.setitem(ns, 'RUN_TRAINING', True)
    monkeypatch.setitem(ns, 'T4_BATCH_PER_GPU', 0)
    monkeypatch.setitem(ns, 'train_records', ns['pd'].DataFrame(
        dict(file_name=['one.jpeg'], image_id=[1])))
    monkeypatch.setitem(ns, 'coco', SimpleNamespace(getAnnIds=lambda **kwargs: list(range(12))))
    monkeypatch.setitem(ns, 'YOLO', lambda *args: SimpleNamespace(model=FakeNetwork()))
    monkeypatch.setattr(ns['torch'].cuda, 'get_device_name', lambda i: 'Tesla T4')
    monkeypatch.setattr(ns['torch'].cuda, 'mem_get_info', lambda i: ((14 if i == 0 else 12) * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(ns['torch'].cuda, 'device', lambda i: nullcontext())
    monkeypatch.setattr(ns['torch'].cuda, 'empty_cache', lambda: None)
    def probe(model, **kwargs):
        events['kwargs'] = kwargs
        if warning_kind == 'rank':
            # Reproduce the actual NumPy warning from one usable profiling point.
            np.polyfit([1.0], [2.0], deg=1)
        elif warning_kind == 'other':
            warnings.warn('Unrelated profiler warning', UserWarning)
        return estimate
    monkeypatch.setattr(autobatch, 'check_train_batch_size', probe)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        batch, plan = ns['choose_dual_t4_batch']()
    assert events['device'] == 'cuda:1'
    assert events['kwargs']['dataset_size'] == 4 and events['kwargs']['amp'] is True
    assert events['kwargs']['batch'] == .70 and events['kwargs']['max_num_obj'] == 12
    assert batch == expected_per_gpu * 2 and batch % 2 == 0
    assert plan['per_gpu'] == expected_per_gpu
    if warning_kind == 'rank':
        assert plan['reason'] == 'ill-conditioned AutoBatch fit'
        assert 'poorly conditioned' in plan['profile_warnings'][0]
        assert not notices  # Handled explicitly, with provenance in the batch plan.
    elif warning_kind == 'other':
        assert any('Unrelated profiler warning' in str(notice.message) for notice in notices)


def test_cpu_prefetch_fallback_preserves_uint8_and_counts_batches(ns):
    runtime = ns['solar_runtime']
    batches = [{'img': ns['torch'].full((2, 3, 16, 16), i, dtype=ns['torch'].uint8)} for i in [1, 2, 3]]
    loader = runtime.PrefetchLoader(batches, 'cpu', cuda_prefetch=True)
    output = list(loader)
    assert len(output) == 3 and loader.stats['images'] == 6
    assert loader.stats['batches'] == 3 and not loader.stats['cuda_prefetch']
    assert [int(batch['img'].max()) for batch in output] == [1, 2, 3]


def test_cuda_prefetch_schedules_next_copy_without_normalizing_inputs(ns, monkeypatch):
    """Check stream dependencies using stand-ins; no local GPU performance tuning."""
    events = []
    class Tensor:
        def __init__(self, name): self.name = name
        def to(self, device, non_blocking):
            assert non_blocking
            events.append('copy_' + self.name)
            return self
        def record_stream(self, stream): events.append('record_' + self.name)
        def __len__(self): return 2
    class Stream:
        def wait_stream(self, other): events.append('wait')
    fake = SimpleNamespace(
        Tensor=Tensor, device=lambda device: SimpleNamespace(type='cuda'),
        cuda=SimpleNamespace(Stream=lambda **kwargs: Stream(), stream=lambda value: nullcontext(),
                             current_stream=lambda device: Stream()))
    runtime = ns['solar_runtime']
    monkeypatch.setattr(runtime, 'torch', fake)
    batches = [{'img': Tensor('first')}, {'img': Tensor('second')}]
    iterator = iter(runtime.PrefetchLoader(batches, 'cuda:0'))
    first = next(iterator)
    assert first['img'].name == 'first'
    assert events == ['copy_first', 'wait', 'record_first', 'copy_second']
    assert next(iterator)['img'].name == 'second'
    with pytest.raises(StopIteration): next(iterator)


def test_nonzero_rank_does_not_write_shared_throughput_log(ns, monkeypatch, tmp_path):
    runtime = ns['solar_runtime']
    monkeypatch.setattr(ns['torch'].distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(ns['torch'].distributed, 'get_rank', lambda: 1)
    trainer = SimpleNamespace(save_dir=tmp_path, train_loader=SimpleNamespace(stats={'batches': 1}))
    runtime.log_training_throughput(trainer)
    assert not (tmp_path / 'input_throughput.csv').exists()


def test_native_real_image_matches_preprocessing_notebook(ns):
    path = ROOT / 'MAGFilo_1.0_Kaggle_2026/train/train_images/20160617133654Bh.jpeg'
    if not path.exists():
        pytest.skip('Local competition sample not available')
    runtime = ns['solar_runtime']
    image = runtime.load_grayscale(path, ns['PREPROCESSING_CONFIG'])
    bgr, auxiliary = runtime.preprocess_model_input(image, ns['PREPROCESSING_CONFIG'], return_aux=True)
    stages = runtime.preprocess_image(image, ns['PREPROCESSING_CONFIG'])
    assert bgr.shape == (2048, 2048, 3)
    assert np.array_equal(bgr[..., ::-1], stages['multi_channel'])
    assert np.array_equal(auxiliary['candidates'], stages['candidates'] * 255)
    assert 850 < auxiliary['disk']['radius'] < 950


def test_real_coco_preparation_caches_exact_native_input_and_preserves_sources(ns, monkeypatch, tmp_path):
    source_root = ROOT / 'MAGFilo_1.0_Kaggle_2026'
    annotation_path = source_root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    if not annotation_path.exists():
        pytest.skip('Local competition annotations unavailable')
    coco = ns['COCO'](str(annotation_path))
    filename = '20160617133654Bh.jpeg'
    source = source_root / 'train/train_images' / filename
    before = ns['file_sha256'](source)
    ids = [image_id for image_id, row in coco.imgs.items() if row['file_name'] == filename]
    workspace = tmp_path / 'yolo_filament_dataset'
    for part in ['images', 'labels']:
        (workspace / part / 'train').mkdir(parents=True)
    for key, value in dict(coco=coco, TRAIN_IMAGE_DIR=source.parent, YOLO_DATA_DIR=workspace).items():
        monkeypatch.setitem(ns, key, value)
    stats = ns['prepare_record'](({'file_name': filename, 'image_ids': ids}, 'train'))
    assert stats['annotators'] == len(ids) and stats['training_instances'] > 0
    assert stats['image_storage'] in {'symlink', 'hardlink', 'compressed_source_copy'}
    assert ns['file_sha256'](workspace / 'images/train' / filename) == before
    assert ns['file_sha256'](source) == before
    assert stats['prepared_input']
    prepared, candidates, metadata = ns['prepared_input_paths'](source, 'train')
    assert prepared.is_file() and candidates.is_file() and metadata.is_file()
    expected = ns['solar_runtime'].preprocess_model_input(
        ns['read_gray'](source), ns['PREPROCESSING_CONFIG'], ns['MODEL_INPUT_MODE'])
    assert np.array_equal(np.load(prepared, allow_pickle=False), expected)


def test_spawned_loader_worker_imports_runtime_and_returns_feature_tensors(ns, cfg, tmp_path):
    from ultralytics.cfg import get_cfg
    from torch.utils.data import DataLoader
    _, workspace, yaml = make_workspace(ns, tmp_path, cfg)
    args = get_cfg(overrides=dict(data=str(yaml), task='segment', imgsz=128,
                                  mosaic=0., mixup=0., copy_paste=0.,
                                  hsv_h=0., hsv_s=0., hsv_v=0., bgr=0., augmentations=[]))
    dataset = ns['solar_runtime'].build_streaming_dataset(
        args, str(workspace / 'images/train'), 2,
        {'names': {0: 'filament'}, 'nc': 1, 'channels': 3}, 'train', 32)
    loader = DataLoader(dataset, batch_size=1, num_workers=1,
                        multiprocessing_context='spawn', prefetch_factor=1,
                        collate_fn=dataset.collate_fn)
    iterator = iter(loader)
    batch = next(iterator)
    assert tuple(batch['img'].shape) == (1, 3, 128, 128)
    assert batch['masks'].numel() > 0
    assert ns['solar_runtime'].StreamingSolarSegmentationTrainer.__module__ != '__main__'
    del iterator, loader


def test_trainer_and_final_validator_use_streaming_loader(ns, cfg, tmp_path):
    from ultralytics.cfg import get_cfg
    _, workspace, yaml = make_workspace(ns, tmp_path, cfg)
    args = get_cfg(overrides=dict(data=str(yaml), task='segment', imgsz=128, workers=2))
    trainer = object.__new__(ns['solar_runtime'].StreamingSolarSegmentationTrainer)
    trainer.args = args
    trainer.model = SimpleNamespace(stride=ns['torch'].tensor([32]))
    trainer.data = {'names': {0: 'filament'}, 'nc': 1, 'channels': 3}
    dataset = trainer.build_dataset(str(workspace / 'images/train'), 'train', 2)
    assert isinstance(dataset, ns['solar_runtime'].StreamingSolarDataset)
    trainer.test_loader, trainer.save_dir, trainer.callbacks = None, workspace, {}
    validator = trainer.get_validator()
    validator.data = trainer.data
    validator.stride = 32
    validation = validator.build_dataset(str(workspace / 'images/val'), batch=1)
    assert isinstance(validation, ns['solar_runtime'].StreamingSolarDataset)
    assert not validation.cache and not validation.augment


def test_d4_transforms_all_feature_channels_without_swapping_them(ns):
    rgb = np.stack([np.arange(49, dtype=np.uint8).reshape(7, 7) + k for k in [0, 50, 100]], axis=-1)
    for k, flip in ns['D4']:
        transformed = ns['d4_transform'](rgb, k, flip)
        restored = ns['d4_inverse_maps'](transformed.transpose(2, 0, 1), k, flip).transpose(1, 2, 0)
        assert np.array_equal(rgb, restored)


def test_predict_engine_receives_bgr_features_and_preserves_shape(ns, monkeypatch):
    image = np.stack([np.full((8, 8), n, np.uint8) for n in [20, 70, 120]], axis=-1)
    canonical = np.arange(64, dtype=np.uint8).reshape(8, 8)
    class FakeModel:
        count = 0
        def predict(self, source, **kwargs):
            k, flip = ns['D4'][self.count]
            self.count += 1
            assert np.array_equal(source, ns['d4_transform'](image, k, flip))
            data = ns['d4_transform'](canonical, k, flip)[None]
            return [SimpleNamespace(probability_maps=data, single_binary=(data > 30).astype(np.uint8))]
    monkeypatch.setitem(ns, 'CONFIDENCE_CANDIDATES', [0.1])
    monkeypatch.setitem(ns, 'TTA_VIEWS', 8)
    engine = object.__new__(ns['InferenceEngine'])
    engine.lanes = ns['queue'].Queue()
    engine.lanes.put((FakeModel(), ns['torch'].device('cpu')))
    probability, single = engine.predict_maps(image)
    assert probability.shape == single.shape == (1, 8, 8)
    assert np.array_equal(probability[0], canonical)


def setup_completed_run(ns, monkeypatch, tmp_path):
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    data_root = tmp_path / 'original_dataset'
    data_root.mkdir()
    dataset = experiment / 'yolo_filament_dataset'
    dataset.mkdir()
    weights = experiment / 'yolo_runs/train/weights/best.pt'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'preserved model weights')
    (weights.parent.parent / 'results.csv').write_text('epoch,loss\n1,1.0\n')
    (weights.parent.parent / 'results.png').write_bytes(b'preserved training history curve')
    ns['save_json'](dataset / 'streaming_config.json', {'run_fingerprint': 'test'})
    (dataset / 'input_cache/train').mkdir(parents=True)
    np.save(dataset / 'input_cache/train/sun.npy', np.zeros((16, 16, 3), np.uint8))
    original = data_root / 'source.png'
    original.write_bytes(b'original data must survive')
    ns['link_source_image'](original, dataset / 'source.png')
    cache = experiment / 'prediction_cache' / ('a' * 16)
    cache.mkdir(parents=True)
    (cache / 'one.npz').write_bytes(b'temporary maps')
    (cache / 'one.json').write_text('{}')
    selected = {'checkpoint': str(weights), 'weights_sha256': ns['file_sha256'](weights),
                'inference': ns['current_inference_config']()}
    ns['save_json'](experiment / 'selected_pipeline.json', selected)
    (experiment / 'report_metrics.csv').write_text('PQ\n0.5\n')
    for key, value in dict(EXPERIMENT_DIR=experiment, DATA_ROOT=data_root, YOLO_DATA_DIR=dataset,
                           PREVIEW_DIR=experiment / 'previews', RUNS_DIR=experiment / 'yolo_runs',
                           run_dir=weights.parent.parent, RUN_FINGERPRINT='test', GPU_IDS=[],
                           CLEANUP_INTERMEDIATES=True, RUN_TRAINING=True, CREATE_SUBMISSION=False,
                           SAVE_PREVIEW_IMAGES=False).items():
        monkeypatch.setitem(ns, key, value)
    return experiment, dataset, original, weights


def test_cleanup_only_after_success_keeps_sources_weights_history_and_metrics(ns, monkeypatch, tmp_path):
    experiment, dataset, original, weights = setup_completed_run(ns, monkeypatch, tmp_path)
    summary = ns['cleanup_completed_run']()
    assert summary['completed']
    assert summary['removed_input_cache_bytes'] > 0
    assert not dataset.exists() and not (experiment / 'prediction_cache').exists()
    assert original.read_bytes() == b'original data must survive'
    assert weights.read_bytes() == b'preserved model weights'
    assert (weights.parent.parent / 'results.csv').is_file()
    assert (weights.parent.parent / 'results.png').is_file()
    assert (experiment / 'report_metrics.csv').is_file()


def test_cleanup_requires_verified_submission_and_rejects_outside_paths(ns, monkeypatch, tmp_path):
    experiment, dataset, original, weights = setup_completed_run(ns, monkeypatch, tmp_path)
    monkeypatch.setitem(ns, 'CREATE_SUBMISSION', True)
    with pytest.raises(RuntimeError, match='Submission audit missing'):
        ns['cleanup_completed_run']()
    assert dataset.exists() and original.exists() and weights.exists()
    monkeypatch.setitem(ns, 'CREATE_SUBMISSION', False)
    monkeypatch.setitem(ns, 'YOLO_DATA_DIR', tmp_path / 'original_dataset')
    with pytest.raises(ValueError, match='unexpected generated directory'):
        ns['cleanup_completed_run']()
    assert dataset.exists() and original.exists()


def test_native_submission_serialization_and_nonoverlap_audit(ns, tmp_path):
    mask = np.zeros((2048, 2048), np.uint8)
    mask[500:520, 500:700] = 1
    rows = ns['submission_rows_for_source']('sun', [ns['encode_mask'](mask)])
    csv_path = tmp_path / 'submission.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = ns['csv'].writer(stream)
        writer.writerow(['filament_id', 'segmentation_rle'])
        writer.writerows(rows)
    audit = ns['audit_submission'](csv_path, {'sun': 1, 'empty': 0})
    assert audit['non_overlapping_masks'] and audit['zero_prediction_images'] == 1
    with csv_path.open('a', newline='', encoding='utf-8') as stream:
        ns['csv'].writer(stream).writerow(['sun_2', rows[0][1]])
    with pytest.raises(ValueError, match='overlaps'):
        ns['audit_submission'](csv_path, {'sun': 2, 'empty': 0})


def test_streaming_calibration_submission_and_cleanup_share_one_pipeline(ns, monkeypatch, tmp_path):
    """Run actual selection/export/cleanup, with deterministic stand-in model predictions."""
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    data_root = tmp_path / 'source_data'
    train_dir, test_dir = data_root / 'train', data_root / 'test'
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    names = ['calibration', 'report', 'test_a', 'test_b']
    image = synthetic_sun(2048)
    for i, name in enumerate(names):
        ns['cv2'].imwrite(str((train_dir if i < 2 else test_dir) / (name + '.png')), image)
    foreground = np.zeros((2048, 2048), np.uint8)
    foreground[900:920, 700:1000] = 1
    gt = ns['encode_mask'](foreground)
    class FakeEngine:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint
        def predict_maps(self, processed):
            assert processed.shape == (2048, 2048, 3)
            assert processed.dtype == np.uint8
            binary = foreground.copy() if self.checkpoint.stem == 'best' else np.zeros_like(foreground)
            return binary[None] * 255, binary[None]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    checkpoints = [experiment / name for name in ['best.pt', 'last.pt']]
    for path in checkpoints:
        path.write_bytes(b'stand-in checkpoint: ' + path.stem.encode())
    dataset = experiment / 'yolo_filament_dataset'
    dataset.mkdir()
    ns['save_json'](dataset / 'streaming_config.json', {'run_fingerprint': 'test'})
    csv_path = tmp_path / 'submission.csv'
    replacements = dict(EXPERIMENT_DIR=experiment, DATA_ROOT=data_root, TRAIN_IMAGE_DIR=train_dir,
                        TEST_IMAGE_DIR=test_dir, YOLO_DATA_DIR=dataset, PREVIEW_DIR=experiment / 'previews',
                        RUNS_DIR=experiment / 'yolo_runs', RUN_FINGERPRINT='test',
                        checkpoint_candidates=checkpoints, InferenceEngine=FakeEngine,
                        CONFIDENCE_CANDIDATES=[.1], PREDICTION_CACHE_POLICY='stream',
                        PREDICTION_PREVIEW_COUNT=1, SUBMISSION_PATH=csv_path,
                        RUN_TRAINING=False, CREATE_SUBMISSION=True, CLEANUP_INTERMEDIATES=True,
                        SAVE_PREVIEW_IMAGES=False, training_seconds=0., GPU_IDS=[],
                        calibration_records=ns['pd'].DataFrame(dict(file_name=['calibration.png'] * 2, image_id=[1, 2])),
                        report_records=ns['pd'].DataFrame(dict(file_name=['report.png'], image_id=[3])),
                        POSTPROCESS_GRID=[dict(confidence=.1, mode='union_cc', min_area=1, clip_to_disk=False)],
                        target_rles=lambda image_id: [gt], display=lambda *args: None)
    for key, value in replacements.items():
        monkeypatch.setitem(ns, key, value)
    monkeypatch.setattr(ns['plt'], 'show', lambda: None)
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    exec(''.join(notebook['cells'][23]['source']), ns)
    assert Path(ns['FROZEN_SELECTION']['checkpoint']).name == 'best.pt'
    assert ns['report_summary'].iloc[0].PQ == pytest.approx(1)
    assert not list((experiment / 'prediction_cache').rglob('calibration.npz'))
    output_audit = ns['create_submission_parallel'](csv_path)
    assert output_audit['non_overlapping_masks'] and output_audit['test_images_processed'] == 2
    assert output_audit['submitted_filaments'] == 2
    assert output_audit['pipeline'] == ns['FROZEN_SELECTION']
    assert {p.stem for p in (experiment / 'prediction_cache').rglob('*.npz')} == {'report', 'test_a'}
    ns['prediction_preview'](train_dir / 'report.png',
                             ns['cached_predict'](None, train_dir / 'report.png', ns['SELECTED_CACHE_DIR']),
                             [gt], 'validation_predictions', [gt])
    ns['prediction_preview'](test_dir / 'test_a.png',
                             ns['cached_predict'](None, test_dir / 'test_a.png', ns['SELECTED_CACHE_DIR']),
                             [gt], 'test_predictions')
    result = ns['cleanup_completed_run']()
    assert result['completed'] and csv_path.is_file()
    assert all(path.is_file() for path in checkpoints)
    assert not dataset.exists() and not (experiment / 'prediction_cache').exists()
    assert ns['file_sha256'](csv_path) == output_audit['sha256']


def test_one_epoch_training_and_final_validation_smoke(ns, cfg, tmp_path):
    """Exercise real backward passes and final best.pt validation with no downloads."""
    _, workspace, yaml = make_workspace(ns, tmp_path, cfg)
    model = ns['YOLO']('yolo11n-seg.yaml')
    model.train(trainer=ns['solar_runtime'].StreamingSolarSegmentationTrainer,
                data=str(yaml), pretrained=False, epochs=1, imgsz=128, batch=2,
                workers=0, device='cpu', cache=False, amp=False, plots=False,
                project=str(tmp_path / 'runs'), name='smoke', optimizer='SGD',
                mosaic=0., mixup=0., copy_paste=0., hsv_h=0., hsv_s=0., hsv_v=0.,
                bgr=0., augmentations=[], verbose=False, deterministic=True)
    assert (Path(model.trainer.save_dir) / 'weights/best.pt').is_file()
    assert (Path(model.trainer.save_dir) / 'weights/last.pt').is_file()
    assert (Path(model.trainer.save_dir) / 'results.csv').is_file()
    assert isinstance(model.trainer.validator, ns['solar_runtime'].StreamingSolarSegmentationValidator)
    assert isinstance(model.trainer.train_loader.dataset, ns['solar_runtime'].StreamingSolarDataset)
    assert all(image is None for image in model.trainer.train_loader.dataset.ims)
    assert len(list((workspace / 'input_cache').rglob('*.npy'))) == 4
    assert (Path(model.trainer.save_dir) / 'input_throughput.csv').is_file()
