"""Build one uploadable Kaggle notebook; builder/source files are NOT runtime inputs."""
import ast
import copy
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
source = (HERE / "pq50_runtime.py").read_text(encoding="utf8")
tree = ast.parse(source)
discovery = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "discover_root")
lines = source.splitlines(keepends=True)
lines[discovery.lineno - 1:discovery.end_lineno] = ['''def discover_root(configured=""):
    """Read the attached competition data exclusively under /kaggle/input."""
    input_root = INPUT_ROOT.resolve()
    name = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if not input_root.is_dir():
        raise FileNotFoundError("/kaggle/input is missing. Run this notebook on Kaggle and attach the competition data.")
    if configured:
        candidates = [Path(configured).resolve()]
        if not candidates[0].is_relative_to(input_root):
            raise ValueError("DATA_ROOT_OVERRIDE must point inside /kaggle/input.")
    else:
        candidates = sorted({p.parent.parent for p in input_root.rglob(name)})
    matches = [root for root in candidates if
               (root / "train" / name).is_file() and
               (root / "train" / "train_images").is_dir() and
               (root / "test" / "test_images").is_dir()]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError("Multiple competition copies found. Set DATA_ROOT_OVERRIDE to one of: " +
                         ", ".join(map(str, matches)))
    raise FileNotFoundError("Attach the filament-segmentation-2026 competition dataset using Add Input. "
                            "Expected its train annotation JSON, train/train_images and test/test_images.")
''']
source = "".join(lines)
source = source.replace('Embedded verbatim by build_pq50_notebook.py. No external filament annotations,',
                        'All code is included in this Kaggle notebook. No external filament annotations,')
source = source.replace('output_root="pq50_outputs", run_name="native_hybrid_v1"',
                        'output_root="/kaggle/working", run_name="kaggle_native_hybrid_v1"')
source = source.replace('VERSION = "pq50-native-hybrid-v1"',
                        'VERSION = "kaggle-native-hybrid-v1"\nINPUT_ROOT = Path("/kaggle/input")\nWORKER_SCRIPT = Path("/kaggle/working/filament_gpu_worker.py")')
source = source.replace('file_sha(__file__)', 'file_sha(WORKER_SCRIPT)')
source = source.replace('Path(__file__)', 'Path(WORKER_SCRIPT)')
# The notebook defines main() but only the generated worker invokes it.
guard = next(n for n in ast.parse(source).body if isinstance(n, ast.If) and
             isinstance(n.test, ast.Compare) and ast.unparse(n.test) == "__name__ == '__main__'")
source = "\n".join(source.splitlines()[:guard.lineno - 1]).rstrip() + "\n"

markers = [
    ("imports_and_io", None, "Imports, configuration and mask I/O"),
    ("data", "def normalize_solar(", "Solar normalization and Kaggle data preparation"),
    ("models", "def device_setup(", "Native mask models, crops and losses"),
    ("metrics", "def update_ema(", "Native prediction and official PQ scoring"),
    ("mask_training", "def train_dense(", "Distributed mask training worker"),
    ("detector_training", "def train_detector(", "Detector training and GPU launchers"),
    ("prediction", "def box_iou(", "Proposal fusion and two-GPU prediction"),
    ("candidates", "FEATURES = ", "Instance candidates and quality features"),
    ("calibration", "def quality_estimator(", "Quality learning and pipeline selection"),
    ("submission", "def evaluate_report(", "Report evaluation and submission export"),
]
offsets = [0 if marker is None else source.index("\n" + marker) + 1 for _, marker, _ in markers]
sections = [(name, title, source[start:end].strip()) for (name, _, title), start, end in
            zip(markers, offsets, offsets[1:] + [len(source)])]

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''
# Solar filaments: complete Kaggle training and submission notebook

**This one notebook contains the complete pipeline.** Its only attached input is
the [competition dataset](https://www.kaggle.com/competitions/filament-segmentation-2026/data).
It installs packages, downloads generic pretrained models, prepares data, trains,
validates and writes **`/kaggle/working/submission.csv`**.

1. Import this `.ipynb` into a new Kaggle notebook.
2. Use **Add Input** to attach the competition data.
3. Set **Accelerator: GPU T4 x2** and **Internet: On**.
4. Run all cells in order. Full training is the default.

Models: YOLO11m box proposals plus ResNet34 and EfficientNetV2-S U-Nets learning
native-resolution masks. It uses two GPUs, **four total loader workers**, separate
annotator targets, grouped validation, learned quality and adaptive mask selection.
The 0.50 score is a target; this new pipeline has no established full-run score.

All helper code is visible below. The notebook automatically assembles its own
worker script for multi-GPU subprocesses. No repository, local Python module,
results archive, previous submission or manually uploaded checkpoint is required.
''')

code('''
import importlib.metadata as metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

# Keep Kaggle's installed CUDA torch/torchvision.
requirements = {
    'ultralytics': ('ultralytics', '8.4.198'),
    'segmentation-models-pytorch': ('segmentation-models-pytorch', '0.5.0'),
    'timm': ('timm', '1.0.30'),
    'scikit-learn': ('scikit-learn', '1.9.1'),
}
install = []
for distribution, (package, version) in requirements.items():
    try:
        installed = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        install.append(f'{package}=={version}')
for module, package in {
    'pycocotools': 'pycocotools', 'scipy': 'scipy', 'cv2': 'opencv-python-headless',
    'pandas': 'pandas', 'matplotlib': 'matplotlib', 'joblib': 'joblib', 'yaml': 'pyyaml',
}.items():
    if importlib.util.find_spec(module) is None:
        install.append(package)
if install:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-cache-dir', *install])
print('Packages ready. Use a fresh kernel if you previously imported different versions.')
''')

md('''
## All helper code

The following cells both define functions for this notebook and capture the exact
same source for GPU workers. Editing and rerunning a helper cell updates its
captured source. Run the worker-assembly cell after all helper cells.
''')

code('''
_FILAMENT_SECTIONS = {}

def _filament_section(line, cell):
    name = line.strip()
    if not name or ' ' in name:
        raise ValueError('Give the helper section a single name.')
    compiled = compile(cell, f'<filament:{name}>', 'exec')
    exec(compiled, get_ipython().user_ns)
    _FILAMENT_SECTIONS[name] = cell

get_ipython().register_magic_function(_filament_section, magic_kind='cell', magic_name='filament_section')
print('Inline helper cells enabled.')
''')

for name, title, text in sections:
    md("### " + title)
    code(f"%%filament_section {name}\n{text}\n")

md('''
## Assemble the GPU worker automatically

The generated file is an output of this notebook. Both GPU ranks run the same
captured helper code. It is recreated by Run All and needs no separate upload.
''')
names = [name for name, _, _ in sections]
code(f'''
_SECTION_ORDER = {names!r}
_missing = [name for name in _SECTION_ORDER if name not in _FILAMENT_SECTIONS]
if _missing:
    raise RuntimeError('Run every helper cell first: ' + ', '.join(_missing))
_worker_source = (chr(10) * 2).join(_FILAMENT_SECTIONS[name].strip() for name in _SECTION_ORDER)
_worker_source += chr(10) * 2 + 'if __name__ == "__main__":' + chr(10) + '    main()' + chr(10)
compile(_worker_source, str(WORKER_SCRIPT), 'exec')
WORKER_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
_temporary_worker = WORKER_SCRIPT.with_suffix('.tmp')
_temporary_worker.write_text(_worker_source, encoding='utf8')
os.replace(_temporary_worker, WORKER_SCRIPT)
print('Created GPU worker:', WORKER_SCRIPT)
''')

md('''
## Run settings

`MODE='full'` trains and generates the real submission. `MODE='smoke'` runs a tiny
environment check and creates only `smoke_submission.csv`. Full mode also refits
on all supplied training labels after freezing validation choices. It is a
substantial training run; total time on two T4s has not been measured.

Model initialization is downloaded automatically: YOLO11m from Ultralytics and
generic ImageNet encoders through SMP/timm. No competition-specific weights are
needed. To reduce memory use, lower a batch size while keeping image sizes fixed.
''')

code('''
MODE = 'full'
DATA_ROOT_OVERRIDE = ''       # normally leave empty; search only /kaggle/input
GPU_IDS = [0, 1]
CPU_WORKERS = 4               # total: two loader workers per GPU
REFIT_FULL = True

CFG = default_config()
CFG.update(mode=MODE, data_root=DATA_ROOT_OVERRIDE, gpu_ids=GPU_IDS,
           cpu_workers=CPU_WORKERS, refit_full=REFIT_FULL,
           output_root='/kaggle/working', run_name='kaggle_native_hybrid_v1',
           detector_weights='yolo11m.pt', pretrained=True)

if not torch.cuda.is_available() or not GPU_IDS or max(GPU_IDS) >= torch.cuda.device_count():
    raise RuntimeError('Select GPU T4 x2 in Kaggle notebook settings before running training.')
print('GPUs:', [(i, torch.cuda.get_device_name(i)) for i in GPU_IDS])
print('CPU loader workers total:', CPU_WORKERS)
print('Mode:', MODE, '| Full-data refit:', REFIT_FULL)
print('Detector initializer:', CFG['detector_weights'])
print('Mask encoders:', [m['encoder'] for m in CFG['dense_models']])
''')

# Retain the already-tested train/validate/export flow, with all helpers now in
# this notebook's namespace rather than imported from a companion module.
base = nbf.read(HERE / "solar-filament-native-hybrid-pq50.ipynb", as_version=4)
for cell in base.cells[5:18]:
    cell = copy.deepcopy(cell)
    if cell.cell_type == "code":
        cell.source = cell.source.replace("pipeline.", "")
    cell.id = nbf.v4.new_code_cell().id
    cells.append(cell)

md('''
## Outputs and validation

The completed full run writes **`/kaggle/working/submission.csv`**. Find it in
Kaggle's Output/Files panel. Its run folder also contains calibration choices,
held-out report metrics, training curves, model weights, prediction previews and
`submission_audit.json`. The exporter checks all 180 test images, native 2048 masks,
unique IDs, compressed COCO RLE round trips and non-overlapping instances.

Training uses the supplied competition annotations. External solar-filament labels
and test annotations are excluded. Generic COCO/ImageNet initialization is used.
The report is evaluated before the optional full-data refit, so its score belongs
to the earlier models. No score above 0.50 is claimed for this notebook.

Method references: [competition evaluation](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates),
[organizer scoring](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook),
[detector/native-mask hybrid](https://github.com/ameyypawar/filament-seg),
[detection/refinement experiments](https://github.com/HeShen-1/filament-segmentation-2026).
Dependency licenses, including Ultralytics AGPL-3.0, continue to apply.
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
})
nbf.validate(notebook)
target = HERE / "solar-filament-kaggle-standalone.ipynb"
nbf.write(notebook, target)
print(target)
