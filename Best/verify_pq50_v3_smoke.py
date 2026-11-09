"""Local CUDA integration smoke; does NOT measure Kaggle throughput or PQ gains."""
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("filament_v3_smoke", HERE/"pq50_v3_runtime.py")
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)

if __name__ == "__main__":
    before = time.time()
    r.INPUT_ROOT = ROOT
    r.WORKER_SCRIPT = HERE/"pq50_v3_runtime.py"
    cfg = r.default_config()
    cfg.update(mode="smoke", gpu_ids=[0], pretrained=False, data_root=str(ROOT/"MAGFilo_1.0_Kaggle_2026"),
        output_root=str(HERE/"analysis_2026_11_08"/"smoke"), run_name="v3_cuda_smoke")
    cfg, manifest = r.prepare(cfg)
    r.prepare_instance_targets(cfg, manifest)
    r.audit_split_leakage(cfg, manifest)
    r.train_all(cfg)
    cal = r.predict_part(cfg, "calibration")
    candidates = r.build_candidate_cache(cfg, "calibration", cal)
    recipe = r.calibrate(cfg, candidates)
    report_maps = r.predict_part(cfg, "report")
    report_candidates = r.build_candidate_cache(cfg, "report", report_maps)
    report = r.evaluate_report(cfg, report_candidates, recipe)
    report = r.add_report_diagnostics(cfg, report_candidates, recipe)
    test_maps = r.predict_part(cfg, "test")
    test_candidates = r.build_candidate_cache(cfg, "test", test_maps)
    # Also exercise the separate all-data refit after preserving the validated CSV.
    cfg["refit_full"] = True
    r.save_json(Path(cfg["run"])/"config.json", cfg)
    submission, variants = r.finish_submission_run(cfg, test_candidates, recipe)
    assert variants["refit"] and Path(variants["validated"]["file"]).is_file()
    summary = dict(kind="local single-GPU integration smoke", elapsed_seconds=time.time()-before,
        gpu=r.torch.cuda.get_device_name(0), run=cfg["run"], submission=str(submission),
        report_PQ=report["PQ"], score_interpretation="one tiny epoch, random initialization; no quality claim",
        validated_and_refit_csvs_preserved=True,
        kaggle_dual_T4_full_run_verified=False)
    r.save_json(HERE/"analysis_2026_11_08"/"v3_smoke_result.json", summary)
    print(json.dumps(summary, indent=2))
