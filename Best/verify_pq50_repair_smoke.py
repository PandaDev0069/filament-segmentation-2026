"""Exercise the actual corrected train/selection/report/export subprocess flow."""
from pathlib import Path
import pq50_runtime as r
from verify_pq50_models import check_cuda
import torch


if __name__ == '__main__':
    root = Path(__file__).resolve().parent / 'repair_2026_11_04' / 'validation'
    manifests = sorted(root.glob('numerics_native_*/manifest.json'), key=lambda p: p.stat().st_mtime)
    if not manifests:
        raise RuntimeError('Run verify_pq50_repair.py first to prepare the smoke inputs.')
    check_cuda(root / 'batch2_fp32.json', pretrained=True, batch=2)
    torch.cuda.empty_cache()
    cfg = r.load_json(manifests[-1])['config']
    r.train_all(cfg)
    calibration = r.predict_part(cfg, 'calibration')
    r.select_dense_members(cfg, calibration)
    candidates = r.build_candidate_cache(cfg, 'calibration', calibration)
    recipe = r.calibrate(cfg, candidates)
    report = r.predict_part(cfg, 'report')
    report_candidates = r.build_candidate_cache(cfg, 'report', report)
    r.evaluate_report(cfg, report_candidates, recipe)
    test = r.predict_part(cfg, 'test')
    test_candidates = r.build_candidate_cache(cfg, 'test', test)
    csv = r.export_submission(cfg, test_candidates, recipe)
    r.save_json(root / 'subprocess_smoke.json', dict(
        completed=True, csv=str(csv), sha256=r.file_sha(csv), mode=cfg['mode'],
        submission_audit=r.load_json(Path(cfg['run'])/'submission_audit.json'),
        note='Short mechanical regression test, not a competition submission or PQ benchmark.'))
    print('Corrected subprocess smoke completed:', csv, flush=True)
