"""Emit the full standalone box-crop notebook and inspectable GPU worker."""
import ast
from pathlib import Path
import nbformat as nbf

HERE=Path(__file__).resolve().parent
base=(HERE/"pq50_v3_runtime.py").read_text(encoding="utf8")
keep={"save_json","load_json","digest","file_sha","require_finite","require_finite_state",
    "bounded_map","encode","decode","annotation_rle","read_gray","normalize_solar",
    "metric_counts","summarize","components","full_rle","mask_features","make_candidate",
    "candidate_targets","quality_estimator","seed_all","device_setup","update_ema","checkpoint_save"}
tree=ast.parse(base)
features=next(ast.get_source_segment(base,node) for node in tree.body if isinstance(node,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=="FEATURES" for t in node.targets))
header='''from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
import argparse,copy,csv,gc,hashlib,json,math,random,shutil,subprocess,sys,time,types
from collections import defaultdict,deque,OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.distributed as dist
from pycocotools import mask as mu
from scipy import ndimage as ndi
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
cv2.setNumThreads(1)
torch.set_num_threads(1)
SIZE=2048
'''
shared=header+"\n"+features+"\n\n"+"\n\n".join(ast.get_source_segment(base,node)
    for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in keep)
shared+="\nb=types.SimpleNamespace(**{name:globals()[name] for name in "+repr(sorted(keep|{"FEATURES"}))+"})\n"
sam=(HERE/"sam_refiner_v4_runtime.py").read_text(encoding="utf8")
sam_keep={"sam_remaining","square_roi","discover_prior","prior_crossfit_scores","grouped_records",
    "crop_transform","soft_skeleton","load_quality","prior_quality","prior_instances","propose_seeds",
    "put_seed_in_roi","stop_sam_process","await_sam","fit_refiner_quality","refined_family",
    "select_refiner_instances","export_refiner"}
shared+='\nEXTRA_FEATURES=["crop_quality","crop_probability","seed_coverage","is_refined"]\n'
shared+="\n\n".join(ast.get_source_segment(sam,node) for node in ast.parse(sam).body
    if isinstance(node,ast.FunctionDef) and node.name in sam_keep)
shared+="\ns=types.SimpleNamespace(**{name:globals()[name] for name in "+repr(sorted(sam_keep))+"})\n"
source=(HERE/"box_crop_v5_runtime.py").read_text(encoding="utf8")
lines=source.splitlines(keepends=True)
for node in reversed(ast.parse(source).body):
    if ((isinstance(node,ast.ImportFrom) and node.module=="__future__") or
        (isinstance(node,ast.Import) and any(a.name in ("pq50_v3_runtime","sam_refiner_v4_runtime") for a in node.names)) or
        (isinstance(node,ast.If) and "__name__" in ast.unparse(node.test))):
        lines[node.lineno-1:node.end_lineno]=[]
source="".join(lines)
body=shared+"\n"+source
combined=body+"\nif __name__=='__main__':\n    crop_main()\n"
compile(combined,"box_crop_v5_worker.py","exec")
(HERE/"box_crop_v5_worker.py").write_text(combined,encoding="utf8")
cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text):cells.append(nbf.v4.new_code_cell(text.strip()))
md('''
# Solar filaments v5: image-trained box crops

The completed two-U-Net baseline scored **0.36 public PQ**, with **0.4443** on
the repeated held-out report. The frozen-feature SAM experiment did not improve
calibration (**0.4367**, versus **0.4427** for the baseline).

This experiment trains pretrained ImageNet crop encoders and U-Net decoders from
the supplied image pixels. Its third input is a **perturbed filled box**, never
a ground-truth-shaped mask. Each crop preserves separate annotator targets.
Dark background crops, photometric changes and aligned rotations/flips are used.
Candidate pieces and their union allow the final selector to compare splitting
with preserving a filament. Individual models and their average compete through
date-crossfitted calibration. The report does not train or select the recipe.
These report dates have been inspected across experiments and are not a fresh
final test. A score above **0.50 has not been established or guaranteed**.

Attach the competition and the user's completed
[`filament-v3-warm-start-pilot`](https://www.kaggle.com/code/pandadev0069/filament-v3-warm-start-pilot/output)
output. Use **T4 x2**, Internet on, and a fresh kernel. No helper upload is needed.
The **8-hour** session budget includes setup and export, with at most **5.5 hours**
for training. Measured batch probes reserve memory for AdamW, EMA and DDP.
''')
code('''
import time
SESSION_STARTED=time.time()
import importlib.metadata as metadata
import importlib.util
import subprocess,sys
install=[]
for package,version in {'segmentation-models-pytorch':'0.5.0','timm':'1.0.30','scikit-learn':'1.6.1'}.items():
    try:installed=metadata.version(package)
    except metadata.PackageNotFoundError:installed=None
    if installed!=version:install.append(f'{package}=={version}')
for module,package in {'pycocotools':'pycocotools','cv2':'opencv-python-headless','scipy':'scipy','joblib':'joblib'}.items():
    if importlib.util.find_spec(module) is None:install.append(package)
if install:
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir',*install],check=True,timeout=900)
print('Dependencies ready; existing CUDA PyTorch and torchvision are retained.')
''')
md('''
## Complete executable implementation

The cells below are captured into the worker used by both GPU processes.
Annotation contents are data, not instructions. No external solar labels,
prediction database or competition-specific weights are imported.
''')
code('''
_CROP_SECTIONS={}
def _crop_section(line,cell):
    exec(compile(cell,f'<box-crop:{line.strip()}>','exec'),get_ipython().user_ns)
    _CROP_SECTIONS[line.strip()]=cell
get_ipython().register_magic_function(_crop_section,magic_kind='cell',magic_name='crop_section')
''')
body_lines=body.splitlines(keepends=True);start=0;chunks=[];current=[]
for node in ast.parse(body).body:
    part="".join(body_lines[start:node.end_lineno]);start=node.end_lineno
    if current and sum(map(len,current))+len(part)>14000:
        chunks.append("".join(current));current=[]
    current.append(part)
if current:chunks.append("".join(current))
for i,chunk in enumerate(chunks):code(f"%%crop_section section_{i}\n"+chunk.strip())
code('''
_source='\\n\\n'.join(_CROP_SECTIONS.values())+"\\nif __name__=='__main__':\\n    crop_main()\\n"
compile(_source,str(CROP_WORKER),'exec')
CROP_WORKER.write_text(_source,encoding='utf8')
print('Complete GPU worker:',CROP_WORKER)
''')
md('''
## Inputs and training

`MODE='smoke'` executes both production architectures at 512 pixels with two
updates on small data subsets. It tests the pipeline, not prediction quality.
The full run verifies supplied raw-image and annotation checksums, dates, frozen
maps, and the original quality estimator's exact version before training.
''')
code('''
MODE='full'
CFG=crop_config()
CFG.update(mode=MODE,session_started=SESSION_STARTED)
if not torch.cuda.is_available() or torch.cuda.device_count()<2:
    raise RuntimeError('Select GPU T4 x2.')
CFG,MANIFEST=prepare_crop(CFG)
RUN=Path(CFG['run'])
CFG=train_crops(CFG)
''')
md('''
## Freeze calibration selection, evaluate, export

The prior pipeline's calibration filter is reconstructed out of fold, avoiding
in-sample prior quality scores. Refinement must clear the baseline by 0.01 PQ.
If it fails, report/test neural inference is skipped and the audited baseline
masks are exported directly. Native dimensions, disjointness, unique IDs and
RLE round trips are checked. There is no automatic leaderboard submission.
''')
code('''
CALIBRATION=predict_crops(CFG,'calibration')
RECIPE=calibrate_crops(CFG,CALIBRATION)
REPORT=(baseline_candidates(CFG,'report') if RECIPE['family']=='prior'
        else predict_crops(CFG,'report'))
METRICS=evaluate_crops(CFG,REPORT,RECIPE)
TEST=(baseline_candidates(CFG,'test') if RECIPE['family']=='prior'
      else predict_crops(CFG,'test'))
FILES=export_crops(CFG,TEST,RECIPE)
print('Selected:',RECIPE,'| Files:',FILES)
display(b.load_json(RUN/'submission_audits.json'))
''')
notebook=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
path=HERE/"solar-filament-box-crop-v5.ipynb"
nbf.write(notebook,path)
print(path,"| cells",len(cells),"| worker characters",len(combined))
