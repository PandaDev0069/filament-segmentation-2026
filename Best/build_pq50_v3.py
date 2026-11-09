"""Build a standalone notebook and its inspectable runtime from the saved run.

The original executed notebook is never modified. No local helper files need to
be attached to Kaggle: the resulting ipynb includes all required source code.
"""
import ast
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
original = json.loads((HERE / "solar-filament-notebook-latest.ipynb").read_text(encoding="utf8"))
source = "\n\n".join("".join(c["source"]).split("\n", 1)[1] for c in original["cells"]
    if "".join(c.get("source", [])).startswith("%%filament_section"))
changes = (HERE / "pq50_v3_changes.py").read_text(encoding="utf8")
patches = {n.name: ast.get_source_segment(changes, n) for n in ast.parse(changes).body
           if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
lines = source.splitlines(keepends=True)
remove = {"train_detector", "box_iou", "fuse_boxes", "detector_predict", "select_dense_members"}
for node in reversed(ast.parse(source).body):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        if node.name in patches:
            lines[first:node.end_lineno] = [patches.pop(node.name) + "\n"]
        elif node.name in remove:
            lines[first:node.end_lineno] = []
source = "".join(lines)
# Additional definitions may occur after callers; all are defined before execution.
source += "\n\n" + "\n\n".join(patches.values()) + "\n"
source = source.replace('VERSION = "kaggle-native-hybrid-v2"', 'VERSION = "filament-instance-balanced-v3"')
source = source.replace('filament_gpu_worker.py', 'filament_v3_worker.py')
source = source.replace('Train-only, date-grouped detector + native mask learning for MAGFiLO Kaggle.',
                        'Train-only, date-grouped, instance-balanced segmentation for MAGFiLO Kaggle.')
source = source.replace('cfg=cfg, annotations=annotation_sha,',
                        'cfg={k: v for k, v in cfg.items() if k != "session_started"}, annotations=annotation_sha,')
source = source.replace('def prepare(cfg):\n    cfg = copy.deepcopy(cfg)',
    'def prepare(cfg):\n    check_deadline(cfg, "data preparation", reserve=600)\n    cfg = copy.deepcopy(cfg)')
source = source.replace('    def process(s):\n        s = copy.deepcopy(s)',
    '    def process(s):\n        check_deadline(cfg, "data preparation", reserve=600)\n        s = copy.deepcopy(s)')
source = source.replace('        for source in manifest["splits"]["test"]:\n            data = load_json(',
    '        for source in manifest["splits"]["test"]:\n            check_deadline(cfg, "CSV export", reserve=30)\n            data = load_json(')
source = source.replace('("torch", "torchvision", "ultralytics", "segmentation-models-pytorch", "timm", "scikit-learn", "pycocotools")',
                        '("torch", "torchvision", "segmentation-models-pytorch", "timm", "scikit-learn", "pycocotools")')
# Keep smoke truly small, including the training budget/probe path.
source = source.replace('cfg.update(dense_epochs=1, dense_steps=2,',
                        'cfg.update(auto_batch=False, maximum_batch_per_gpu=1, dense_epochs=1, dense_steps=2,')
# Old preview did not depend on detectors; the instance generation is now v3-only.
source += '\nif __name__ == "__main__":\n    main()\n'
compile(source, "pq50_v3_runtime.py", "exec")
(HERE / "pq50_v3_runtime.py").write_text(source, encoding="utf8")

cells = []
def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Solar filaments v3: equal-instance supervision, under a measured session budget

Upload **this notebook alone**, attach the competition dataset, select **GPU T4 x2**,
enable Internet, then Run All. Full mode trains, calibrates, evaluates and writes
`/kaggle/working/submission.csv`. Use a fresh kernel.

For this run, also attach the output of
[`pandadev0069/solar-filament-notebook`](https://www.kaggle.com/code/pandadev0069/solar-filament-notebook/output)
using Add Input > Notebook Output. The pipeline can warm-start from your already
trained U-Nets after verifying exact train/calibration/report partitions,
image hashes and annotation SHA. The new refinement head initially preserves
their predictions exactly. Without that optional input it uses ImageNet weights.

The previous executed run scored 0.4323 on held-out data and 0.36 publicly
(user reported). It trained a detector for 1.925 hours that its final recipe did
not use. This version spends that budget on segmentation: separate ImageNet
ResNet34/EfficientNetV2-S encoders, FP32, measured batch sizes, 768-pixel multiscale
crops, a full-resolution refinement head, and equal-weight instance soft-IoU.

**0.50 public PQ is an objective, not a promised or measured result.** A new full
Kaggle run and leaderboard submission are necessary to measure improvement.
The runtime targets **9.5 elapsed hours from the first cell**, including setup.
It budgets training using measured forward/backward and validation throughput,
reserves time for calibration and inference, and terminates child processes at
the deadline. A deadline failure preserves checkpoints but may leave no complete
submission; it must not be described as a verified sub-10-hour successful run.
""")
code("""
import time
SESSION_STARTED = time.time()
import importlib.metadata as metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

# Retain Kaggle's CUDA-enabled PyTorch and torchvision.
requirements = {'segmentation-models-pytorch': '0.5.0', 'timm': '1.0.30'}
install = []
for package, version in requirements.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        install.append(f'{package}=={version}')
for module, package in {'pycocotools': 'pycocotools', 'sklearn': 'scikit-learn',
    'cv2': 'opencv-python-headless', 'scipy': 'scipy', 'pandas': 'pandas',
    'matplotlib': 'matplotlib', 'joblib': 'joblib'}.items():
    if importlib.util.find_spec(module) is None:
        install.append(package)
if install:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir', *install],
                   check=True, timeout=900)
print('Packages ready. Session timer includes this setup cell.')
""")
md("""
## Included source

These cells define the pipeline and capture exactly the same source for the
importable GPU workers. They require no helper upload. Run them all in order.
The supplied annotations are read as data, never as instructions.
""")
code("""
_FILAMENT_SECTIONS = {}
def _filament_section(line, cell):
    name = line.strip()
    exec(compile(cell, f'<filament-v3:{name}>', 'exec'), get_ipython().user_ns)
    _FILAMENT_SECTIONS[name] = cell
get_ipython().register_magic_function(_filament_section, magic_kind='cell', magic_name='filament_section')
""")
# Split at complete top-level statements, never through a class/function/decorator.
tree = ast.parse(source)
nodes = tree.body[:-1]  # exclude worker-only __main__ guard
lines = source.splitlines(keepends=True)
chunks, current, start = [], [], 0
for node in nodes:
    end = node.end_lineno
    part = "".join(lines[start:end])
    if current and sum(map(len, current))+len(part)>14000:
        chunks.append("".join(current)); current=[]
    current.append(part); start=end
if current:
    chunks.append("".join(current))
for i, chunk in enumerate(chunks):
    names = [n.name for n in ast.parse(chunk).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    md(f"### Source {i+1}: " + ", ".join(names[:5]))
    code(f"%%filament_section section_{i}\n"+chunk.strip())
code("""
_worker_source = '\\n\\n'.join(_FILAMENT_SECTIONS[name].strip() for name in _FILAMENT_SECTIONS)
_worker_source += '\\n\\nif __name__ == "__main__":\\n    main()\\n'
compile(_worker_source, str(WORKER_SCRIPT), 'exec')
WORKER_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
WORKER_SCRIPT.write_text(_worker_source, encoding='utf8')
print('Wrote complete GPU worker:', WORKER_SCRIPT)
""")
md("""
## Configuration and preparation

`MODE='full'` is the real run. `MODE='smoke'` checks the complete path with a tiny
randomly initialized network and writes **smoke_submission.csv** only. Both modes
retain 2048×2048 evaluation masks. Model crop resizing is an explicit augmentation;
held-out and test images are predicted at their original pixel spacing.

The split uses the same seed and date fractions as the latest notebook, allowing
a direct comparison where the input files match. All annotators for a physical
image remain in one partition, and targets stay separate. The report partition
is never used for checkpoint selection, model selection, quality fitting or tuning.
The same report dates are reused across experiments, so repeated architecture
comparisons do not make this a fresh, permanently untouched benchmark.
The cache checks decoded image hashes across partitions before training.
""")
code("""
MODE = 'full'
REFIT_FULL = False  # optional final refit; preserves the validated CSV separately
CFG = default_config()
CFG.update(mode=MODE, gpu_ids=[0, 1], session_started=SESSION_STARTED,
           total_hours=9.5, training_hours=5.6, refit_full=REFIT_FULL)
if MODE == 'smoke':
    CFG['pretrained'] = False
if not torch.cuda.is_available() or max(CFG['gpu_ids']) >= torch.cuda.device_count():
    raise RuntimeError('Select Kaggle GPU T4 x2 before running.')
print('GPUs:', [(i, torch.cuda.get_device_name(i)) for i in CFG['gpu_ids']])
CFG = configure_previous_weights(CFG)
CFG, MANIFEST = prepare(CFG)
verify_previous_split(CFG, MANIFEST)
MANIFEST = prepare_instance_targets(CFG, MANIFEST)
audit_split_leakage(CFG, MANIFEST)
RUN = Path(CFG['run'])
print('Run:', RUN)
print('Hours remaining:', round(deadline_remaining(CFG)/3600, 2))
""")
md("""
## Train both encoders

Jobs run sequentially; each uses both GPUs with DDP. Total loader workers stay at
four. Each encoder receives a real FP32 memory/throughput probe; the slower rank
and smaller safe batch determine the shared plan. Foreground, contour and equal
instance-IoU losses are evaluated in FP32. ImageNet encoder BN statistics stay
frozen; decoder/refinement normalization uses GroupNorm. Finite checks precede
optimizer updates and checkpoint writes. No detector is trained or downloaded.
""")
code("train_all(CFG)\nprint('Completed checkpoints:', model_paths(CFG, all_members=True))")
code("""
import pandas as pd
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for member in range(len(CFG['dense_models'])):
    history = pd.read_json(RUN / 'selection_models' / f'dense_{member}' / 'history.jsonl', lines=True)
    label = CFG['dense_models'][member]['encoder']
    axes[0].plot(history.epoch, history.loss, label=label)
    axes[1].plot(history.epoch, history.fixed_rule_calibration_PQ, 'o-', label=label)
    axes[2].plot(history.epoch, history.peak_vram_gib, label=label)
for ax, title in zip(axes, ['Training loss', 'Checkpoint selection PQ (calibration)', 'Peak training VRAM, GiB']):
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend()
fig.tight_layout()
fig.savefig(RUN / 'training_curves.png', dpi=130)
plt.show()
""")
md("""
## Select the pipeline, then freeze it

Four-view flip TTA uses overlapping native tiles and Hann blending. Calibration
compares each encoder and their mean, five mask thresholds, untouched connected
components and a gentle 3×3 closing. The quality regressor is crossfitted by date.
The original independent annotator masks and the organizer's strict IoU > 0.5
counting logic define the objective. Selection PQ is optimistic because it also
selects checkpoints and recipes; the next independent report is the honest check.

Every candidate family already contains disjoint masks. Calibration reuses their
RLEs rather than allocating and re-encoding full canvases for each trial.
""")
code("""
CALIBRATION_MAPS = predict_part(CFG, 'calibration')
CALIBRATION_CANDIDATES = build_candidate_cache(CFG, 'calibration', CALIBRATION_MAPS)
RECIPE = calibrate(CFG, CALIBRATION_CANDIDATES)
display(pd.DataFrame(load_json(RUN / 'calibration_grid.json')).head(12))
""")
code("""
REPORT_MAPS = predict_part(CFG, 'report')
REPORT_CANDIDATES = build_candidate_cache(CFG, 'report', REPORT_MAPS)
REPORT = evaluate_report(CFG, REPORT_CANDIDATES, RECIPE)
REPORT = add_report_diagnostics(CFG, REPORT_CANDIDATES, RECIPE)
display(pd.DataFrame([{k: REPORT[k] for k in ['PQ', 'SQ', 'RQ', 'TP', 'FP', 'FN', 'date_bootstrap_95ci']}]))
display(pd.DataFrame(REPORT['recall_by_area']).T)
preview(CFG, 'report', REPORT_CANDIDATES, RECIPE)
plt.show()
""")
md("""
## Predict all test images and audit the CSV

The frozen models and recipe first predict the test set. The optional
`REFIT_FULL=True` then refits the selected models on all supplied training labels
at a reduced learning rate, only if its separate time allocation fits. The
validated CSV/audit is preserved before refitting, and the refit receives no
held-out score. Both CSV variants remain reviewable. There is no automatic
leaderboard submission. Export audits every COCO RLE, native
dimensions, unique IDs, test-image coverage and mask non-overlap. Empty predictions
are recorded in the audit; no fabricated masks are inserted for empty images.
""")
code("""
TEST_MAPS = predict_part(CFG, 'test')
TEST_CANDIDATES = build_candidate_cache(CFG, 'test', TEST_MAPS)
SUBMISSION, SUBMISSION_VARIANTS = finish_submission_run(CFG, TEST_CANDIDATES, RECIPE)
preview(CFG, 'test', TEST_CANDIDATES, RECIPE)
plt.show()
elapsed = time.time() - SESSION_STARTED
save_json(RUN / 'session_timing.json', dict(elapsed_seconds=elapsed,
    configured_budget_seconds=CFG['total_hours']*3600, finished_within_budget=elapsed<CFG['total_hours']*3600,
    leaderboard_pq='not measured', submission=str(SUBMISSION)))
from IPython.display import FileLink, display
display(FileLink(str(SUBMISSION)))
display(FileLink(SUBMISSION_VARIANTS['validated']['file']))
if SUBMISSION_VARIANTS['refit']:
    display(FileLink(SUBMISSION_VARIANTS['refit']['file']))
print(f'Total elapsed: {elapsed/3600:.2f} h; held-out PQ: {REPORT["PQ"]:.4f}')
""")
md("""
## Competition scope and reproducibility

Only the supplied **train annotations** supervise learning. Initialization uses
generic ImageNet weights or this user's verified train-only previous checkpoints;
public MAGFiLO labels, test metadata labels,
competition-specific outside checkpoints and other people's submissions are
excluded. Test images are used for image-only preprocessing and inference.
The competition overview allows external sources but prohibits additional
ground-truth metadata for training; test inference uses the supplied H-alpha
images. The Rules and foundational rules were also retrieved through the
authenticated Kaggle CLI. They allow reasonably accessible external tools/models
and prohibit manual labels/predictions on validation or test records.

The run folder includes exact package versions, source/config/image fingerprints,
date partitions, decoded-image duplicate audit, GPU plans, per-epoch histories,
best/last checkpoints, all calibration trials, untouched report metrics with
date-bootstrap intervals, per-size recall, prediction provenance, CSV integrity
audit and session timing. Preserve the folder with the CSV for reproduction.

Sources: [competition overview and evaluation](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates),
[organizer self-evaluation](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook),
[SMP model documentation](https://smp.readthedocs.io/en/latest/models.html).
The established benchmark behavior and limitations are documented in
`analysis_2026_11_08/ANALYSIS.md` in the local project; it is not a runtime dependency.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
    language_info=dict(name="python", version="3.13"),
    kaggle=dict(accelerator="nvidiaTeslaT4", isGpuEnabled=True, isInternetEnabled=True,
        language="python", sourceType="notebook", dataSources=[dict(sourceType="competition", sourceId=143678)]))
nbf.validate(nb)
target = HERE / "solar-filament-instance-balanced-under-10h.ipynb"
nbf.write(nb, target)
print(f"Built {target.name}: {len(cells)} cells; {len(source):,} source characters")
