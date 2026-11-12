"""Behavioral checks for the standalone preprocessing notebook's definition cells."""
from pathlib import Path
import copy
import json

import numpy as np
from PIL import Image
import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / 'Final/solar-filament-pre-processing.ipynb'


@pytest.fixture(scope='module')
def pipeline():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    namespace = {'__name__': 'solar_preprocessing_notebook'}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'{NOTEBOOK.name}:cell-{index}', 'exec')
        if 'pipeline-definitions' in cell.get('metadata', {}).get('tags', []):
            exec(''.join(cell['source']), namespace)
    # Read the actual notebook defaults without running path discovery or previews.
    import ast
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'CONFIG' for target in node.targets):
                namespace['CONFIG'] = ast.literal_eval(node.value)
    return namespace


@pytest.fixture
def cfg(pipeline):
    result = copy.deepcopy(pipeline['CONFIG'])
    result.update(expected_shape=[256, 256], hough_coarse_size=256, hough_fine_size=256,
                  radial_bins=32, radial_sample_stride=2, polynomial_sample_stride=4)
    return result


def synthetic_sun():
    y, x = np.ogrid[:256, :256]
    rho = np.hypot(x - 127.5, y - 127.5) / 110
    disk = rho <= 1
    array = np.where(disk, 190 - 55 * rho ** 2 + 8 * x / 256, 0)
    array[(abs(y - 130) < 2) & (x > 80) & (x < 180)] -= 70
    return np.clip(array, 0, 255).astype(np.uint8)


def test_full_pipeline_preserves_geometry_and_separates_channels(pipeline, cfg):
    image = synthetic_sun()
    unchanged = image.copy()
    result = pipeline['preprocess_image'](image, cfg)
    assert np.array_equal(image, unchanged)
    assert abs(result['metadata']['disk']['radius'] - 110) < 5
    assert abs(result['metadata']['disk']['cx'] - 127.5) < 5
    for stage in ('enhanced', 'corrected', 'laplacian', 'candidates'):
        assert result[stage].shape == image.shape
        assert np.isfinite(result[stage]).all()
        assert np.all(result[stage][~result['disk_mask']] == 0)
    assert result['enhanced'].dtype == np.uint8
    assert result['candidates'].dtype == bool
    assert np.array_equal(result['multi_channel'][..., 0], image)
    assert np.array_equal(result['multi_channel'][..., 1], result['enhanced'])
    assert np.array_equal(result['multi_channel'][..., 2], result['candidates'] * 255)
    assert np.array_equal(result['candidates'], result['filtered_candidates'])
    assert len(result['metadata']['threshold_selection']['counts']) == 20


def test_laplacian_sign_and_threshold_uses_new_interval_evidence(pipeline, cfg):
    image = np.tile(np.linspace(160, 200, 64).astype(np.uint8), (64, 1))
    image[20:40, 20:40] = 120
    image[25:35, 25:35] = 100
    mask = np.ones(image.shape, bool)
    laplacian, eligible, raw, diagnostics = pipeline['adaptive_candidates'](image, mask, cfg)
    assert laplacian[20, 25] > 0  # inside the dark rectangle's edge
    scores = diagnostics['mean_laplacian']
    best = int(np.nanargmax([np.nan if v is None else v for v in scores]))
    assert diagnostics['threshold'] == diagnostics['edges'][best + 1]
    assert np.array_equal(raw, eligible & (image / np.float32(255) < diagnostics['threshold']))
    for i, score in enumerate(scores):
        newly_included = (eligible & (image / np.float32(255) >= diagnostics['edges'][i])
                          & (image / np.float32(255) < diagnostics['edges'][i + 1]))
        assert newly_included.sum() == diagnostics['counts'][i]
        if score is not None:
            assert score == pytest.approx(float(laplacian[newly_included].mean()))
    cfg['threshold_edge'] = 'lower'
    _, eligible, lower_raw, lower_diagnostics = pipeline['adaptive_candidates'](image, mask, cfg)
    assert lower_diagnostics['threshold'] == diagnostics['edges'][best]
    assert np.array_equal(lower_raw, eligible & (image / np.float32(255) < diagnostics['edges'][best]))


def test_flat_candidate_branch_is_finite_and_empty(pipeline, cfg):
    array = np.full((64, 64), 100, np.uint8)
    laplacian, _, mask, diagnostics = pipeline['adaptive_candidates'](
        array, np.ones(array.shape, bool), cfg)
    assert not mask.any()
    assert np.isfinite(laplacian).all()
    assert diagnostics['threshold'] is None
    with pytest.raises(ValueError, match='almost no contrast'):
        pipeline['detect_solar_disk'](array, cfg)


def test_disk_validation_excludes_outer_halo_and_partial_disk(pipeline, cfg):
    assert pipeline['valid_circle'](127.5, 127.5, 110, (256, 256), cfg)
    assert not pipeline['valid_circle'](127.5, 127.5, 130, (256, 256), cfg)
    assert not pipeline['valid_circle'](110, 127.5, 120, (256, 256), cfg)


def test_eight_connectivity_and_optional_closing(pipeline, cfg):
    array = np.zeros((20, 20), bool)
    array[2, 2] = array[3, 3] = True
    array[10:13, 10:13] = True
    cfg.update(min_component_area_px=2, min_component_disk_fraction=0)
    filtered, metadata = pipeline['filter_components'](array, 400, cfg)
    assert metadata['raw_components'] == 2
    assert filtered[2, 2] and filtered[3, 3]
    assert np.array_equal(pipeline['conservative_closing'](filtered, np.ones_like(array), cfg), filtered)
    cfg['closing_kernel'] = 5
    with pytest.raises(ValueError, match='Conservative closing'):
        pipeline['conservative_closing'](filtered, np.ones_like(array), cfg)


def records_for_split():
    return [dict(source=f'train/{name}.jpeg', name=f'{name}.jpeg', output_name=f'{name}.png',
                 source_split='train', group=group, image_ids=ids, coco_shape=[256, 256])
            for name, group, ids in [('20200101000000Ah', '20200101', ['a', 'b']),
                                    ('20200101010000Bh', '20200101', ['c']),
                                    ('20200102000000Ah', '20200102', ['d']),
                                    ('20200103000000Ah', '20200103', ['e'])]]


def test_split_keeps_same_date_and_duplicate_ids_together_and_detects_changes(pipeline, tmp_path):
    records = records_for_split()
    path = tmp_path / 'manifest.json'
    manifest = pipeline['freeze_manifest'](records, path, 42, .3)
    assert manifest['records'][0]['split'] == manifest['records'][1]['split']
    assert manifest['records'][0]['image_ids'] == ['a', 'b']
    assert manifest == pipeline['freeze_manifest'](records, path, 42, .3)
    with pytest.raises(ValueError, match='Saved split differs'):
        pipeline['freeze_manifest'](records, path, 100, .3)


def test_validation_rejects_high_bit_depth_and_wrong_coco_geometry(pipeline, cfg, tmp_path):
    high_bit = tmp_path / 'high.tif'
    Image.fromarray(np.full((256, 256), 1000, np.uint16)).save(high_bit)
    with pytest.raises(ValueError, match='expected uint8'):
        pipeline['load_grayscale'](high_bit, cfg)
    normal = tmp_path / 'normal.png'
    Image.fromarray(synthetic_sun()).save(normal)
    with pytest.raises(ValueError, match='COCO metadata'):
        pipeline['load_grayscale'](normal, cfg, [128, 128])


def test_coco_subset_preserves_all_annotation_fields_and_duplicate_image_ids(pipeline, tmp_path):
    annotations = [dict(id=i, image_id=image_id, segmentation=[[1, 1, 2, 1, 2, 2]],
                        bbox=[1, 1, 1, 1], category_id=1, area=.5,
                        iscrowd=0, spine=[[1, 1], [2, 2]])
                   for i, image_id in enumerate(['a', 'b', 'd'])]
    coco = dict(info={'description': 'test'}, categories=[{'id': 1, 'name': 'filament'}],
                images=[dict(id=i, file_name='source.jpeg', height=256, width=256)
                        for i in ['a', 'b', 'd']], annotations=annotations)
    before = copy.deepcopy(coco)
    output = tmp_path / 'annotations.json'
    record = records_for_split()[0]
    pipeline['export_coco_subset'](coco, [record], output)
    subset = json.loads(output.read_text())
    assert coco == before
    assert subset['annotations'] == annotations[:2]
    assert [r['id'] for r in subset['images']] == ['a', 'b']
    assert all(r['file_name'] == record['output_name'] for r in subset['images'])


def test_lossless_export_rgb_order_and_cache_invalidation(pipeline, cfg, tmp_path):
    source = tmp_path / 'input/train/sun.png'
    source.parent.mkdir(parents=True)
    image = synthetic_sun()
    Image.fromarray(image).save(source)
    record = dict(source='train/sun.png', output_name='sun.png', split='train', image_ids=['a'],
                  coco_shape=[256, 256])
    root = tmp_path / 'cache'
    first = pipeline['cache_record'](record, tmp_path / 'input', root, cfg)
    assert first['status'] == 'processed'
    assert pipeline['cache_record'](record, tmp_path / 'input', root, cfg)['status'] == 'cached'
    original = np.asarray(Image.open(root / 'train/original/sun.png'))
    rgb = np.asarray(Image.open(root / 'train/multi_channel/sun.png'))
    enhanced = np.asarray(Image.open(root / 'train/enhanced/sun.png'))
    candidate = np.asarray(Image.open(root / 'train/candidates/sun.png'))
    assert np.array_equal(original, image)
    assert np.array_equal(rgb[..., 0], original)
    assert np.array_equal(rgb[..., 1], enhanced)
    assert np.array_equal(rgb[..., 2], candidate)
    laplacian = np.load(root / 'train/laplacian/sun.npy', allow_pickle=False)
    assert laplacian.dtype == np.float32 and laplacian.min() < 0 < laplacian.max()
    with np.load(root / 'train/float_stages/sun.npz', allow_pickle=False) as arrays:
        assert set(arrays.files) == {'radial', 'background', 'corrected'}
        assert arrays['corrected'].shape == image.shape
    (root / 'train/enhanced/sun.png').unlink()
    assert pipeline['cache_record'](record, tmp_path / 'input', root, cfg)['status'] == 'processed'
    cfg['clahe_clip_limit'] = 3
    assert pipeline['cache_record'](record, tmp_path / 'input', root, cfg)['status'] == 'processed'


def test_candidate_evaluation_counts_and_rle_round_trip(pipeline):
    mask = np.zeros((32, 32), np.uint8)
    mask[10:20, 10:20] = 1
    rle = pipeline['coco_mask'].encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('ascii')
    annotation = dict(segmentation=rle, id=1, image_id='a', iscrowd=0)
    decoded = pipeline['decode_annotation'](annotation, mask.shape)
    assert np.array_equal(decoded, mask)
    record = dict(source_split='train', split='train', source='train/sun.png', image_ids=['a'])
    result = dict(candidates=decoded, metadata={'preprocessing_seconds': .1})
    metrics, truth = pipeline['evaluate_candidates'](record, result, {'a': [annotation]})
    assert np.array_equal(truth, mask)
    assert metrics['candidate_precision'] == metrics['candidate_recall'] == 1
    assert metrics['iou'] == metrics['dice'] == metrics['component_pq_diagnostic'] == 1
    assert metrics['split_gt_instances'] == metrics['merged_components'] == 0
    record['source_split'] = 'test'
    with pytest.raises(ValueError, match='training annotations'):
        pipeline['evaluate_candidates'](record, result, {'a': [annotation]})


def test_percentile_normalization_ignores_space_and_handles_constants(pipeline):
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    image = np.zeros((4, 4), np.float32)
    image[mask] = [10, 20, 30, 40]
    normalized, stats = pipeline['robust_unit'](image, mask, [0, 100])
    image[~mask] = 1e6
    alternate, alternate_stats = pipeline['robust_unit'](image, mask, [0, 100])
    assert np.array_equal(normalized, alternate)
    assert stats == alternate_stats == {'low': 10., 'high': 40.}
    image[mask] = 50
    constant, _ = pipeline['robust_unit'](image, mask, [1, 99])
    assert np.all(constant[mask] == .5) and np.all(constant[~mask] == 0)


def test_parallel_export_all_splits_reports_failures_and_excludes_failed_labels(pipeline, cfg, tmp_path):
    input_dir, output_dir = tmp_path / 'input', tmp_path / 'output'
    records = []
    images = []
    for split, name in [('train', 'sun_a'), ('val', 'sun_b'), ('test', 'sun_c'), ('train', 'bad')]:
        source_split = 'test' if split == 'test' else 'train'
        source = input_dir / source_split / (name + '.png')
        source.parent.mkdir(parents=True, exist_ok=True)
        if name == 'bad':
            source.write_bytes(b'not an image')
        else:
            Image.fromarray(synthetic_sun()).save(source)
        records.append(dict(source=source.relative_to(input_dir).as_posix(), name=source.name,
                            output_name=source.name, split=split, source_split=source_split,
                            image_ids=[] if split == 'test' else [name], coco_shape=[256, 256]))
        if split != 'test':
            images.append(dict(id=name, file_name=source.name, height=256, width=256))
    coco = dict(images=images, annotations=[], categories=[])
    successes, errors = pipeline['export_dataset'](
        records, input_dir, output_dir, cfg, coco, ('train', 'val', 'test'), workers=2)
    assert len(successes) == 3 and len(errors) == 1
    assert errors[0]['source'] == 'train/bad.png'
    assert not (output_dir / 'test/annotations.json').exists()
    assert [r['id'] for r in json.loads((output_dir / 'train/annotations.json').read_text())['images']] == ['sun_a']
    assert [r['id'] for r in json.loads((output_dir / 'val/annotations.json').read_text())['images']] == ['sun_b']
    assert len(json.loads((output_dir / 'test/image_manifest.json').read_text())) == 1
