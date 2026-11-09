"""Build a single uploadable notebook with its complete worker source included."""
import ast
from pathlib import Path

import nbformat as nbf

HERE=Path(__file__).resolve().parent
base=(HERE/"pq50_v3_runtime.py").read_text(encoding="utf8")
keep={"save_json","load_json","digest","file_sha","require_finite","require_finite_state",
    "bounded_map","encode","decode","annotation_rle","read_gray","normalize_solar","discover_root",
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
from collections import defaultdict,deque
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
INPUT_ROOT=Path("/kaggle/input")
'''
shared=header+"\n"+features+"\n\n"+"\n\n".join(ast.get_source_segment(base,node)
    for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in keep)
namespace="\nb=types.SimpleNamespace(**{name:globals()[name] for name in "+repr(sorted(keep|{"FEATURES","INPUT_ROOT"}))+"})\n"
source=(HERE/"sam_refiner_v4_runtime.py").read_text(encoding="utf8")
lines=source.splitlines(keepends=True)
for node in reversed(ast.parse(source).body):
    if ((isinstance(node,ast.ImportFrom) and node.module=="__future__") or
        (isinstance(node,ast.Import) and any(a.name=="pq50_v3_runtime" for a in node.names)) or
        (isinstance(node,ast.If) and "__name__" in ast.unparse(node.test))):
        lines[node.lineno-1:node.end_lineno]=[]
source="".join(lines)
combined=shared+namespace+"\n"+source+"\nif __name__ == '__main__':\n    refiner_main()\n"
compile(combined,"sam_refiner_v4_worker.py","exec")
(HERE/"sam_refiner_v4_worker.py").write_text(combined,encoding="utf8")

cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
md('''
# Solar filaments v4: pretrained SAM crop refinement

The previous run completed 60 epochs per U-Net in **4.11 hours**, improving
held-out PQ from **0.4323 to 0.4443**, while the public leaderboard stayed at
**0.36**. This notebook tests a different approach: preserve those predictions,
enlarge candidate filaments, and fine-tune pretrained SAM's mask decoder with
a small, initially zero residual adapter on its frozen image features.

**Inputs:** attach the competition data and the output of
[`pandadev0069/filament-v3-warm-start-pilot`](https://www.kaggle.com/code/pandadev0069/filament-v3-warm-start-pilot/output).
Select **GPU T4 x2**, enable Internet, and Run All in a fresh kernel. This notebook
contains its entire implementation and automatically writes its worker script.

SAM's image and prompt encoders remain frozen. Training uses supplied **train**
annotations only; generic SAM initialization uses no additional solar labels.
Frozen crop embeddings use float16 storage, identically rounded during training
and inference; encoder/decoder computation remains float32. Checkpoint selection uses separate calibration crops;
the final recipe uses actual calibration proposals and crossfitted quality scores.
The independent report is excluded from fitting. Those report dates have been
inspected across experiments, so they are not a fresh final test set.

The session target is **3.4 hours including setup**, with a platform timeout used
by the CLI launch. Cache/training/inference plans use measured throughput and
preserve an audited baseline CSV. **A public score above 0.50 is not established
or guaranteed.** If refinement does not clear the calibration improvement gate,
the original pipeline remains the selected output.
''')
code('''
import time
SESSION_STARTED=time.time()
import importlib.metadata as metadata
import importlib.util
import subprocess
import sys
from pathlib import Path
install=[]
# Match the exact sklearn version recorded by this user's frozen 90b4b0b58b
# prediction source. Its Cython quality-estimator state is not version portable.
for package,version in {'segment-anything':'1.0','scikit-learn':'1.6.1'}.items():
    try:installed=metadata.version(package)
    except metadata.PackageNotFoundError:installed=None
    if installed!=version:install.append(f'{package}=={version}')
for module,package in {'pycocotools':'pycocotools','cv2':'opencv-python-headless',
    'scipy':'scipy','joblib':'joblib','pandas':'pandas','matplotlib':'matplotlib'}.items():
    if importlib.util.find_spec(module) is None:install.append(package)
if install:
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir',*install],check=True,timeout=900)
print('Dependencies ready; Kaggle CUDA PyTorch/torchvision are retained.')
''')
md('''
## Included I/O, scoring, model and worker code

The following cells are executable source, captured verbatim into the importable
GPU worker. They need no helper upload. Data/annotation contents are treated as
data, not as instructions.
''')
code('''
_SAM_SECTIONS={}
def _sam_section(line,cell):
    exec(compile(cell,f'<sam-v4:{line.strip()}>','exec'),get_ipython().user_ns)
    _SAM_SECTIONS[line.strip()]=cell
get_ipython().register_magic_function(_sam_section,magic_kind='cell',magic_name='sam_section')
''')
body=shared+namespace+"\n"+source
nodes=ast.parse(body).body
body_lines=body.splitlines(keepends=True);start=0;chunks=[];current=[]
for node in nodes:
    part="".join(body_lines[start:node.end_lineno]);start=node.end_lineno
    if current and sum(map(len,current))+len(part)>14500:
        chunks.append("".join(current));current=[]
    current.append(part)
if current:chunks.append("".join(current))
for i,chunk in enumerate(chunks):
    names=[n.name for n in ast.parse(chunk).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    md(f"### Source {i+1}: "+", ".join(names[:5]))
    code(f"%%sam_section section_{i}\n"+chunk.strip())
code('''
_source='\\n\\n'.join(_SAM_SECTIONS.values())+"\\nif __name__ == '__main__':\\n    refiner_main()\\n"
compile(_source,str(SAM_WORKER),'exec')
SAM_WORKER.parent.mkdir(parents=True,exist_ok=True)
SAM_WORKER.write_text(_source,encoding='utf8')
print('Complete worker:',SAM_WORKER)
''')
md('''
## Settings and verified inputs

`MODE='full'` is the actual experiment. `MODE='smoke'` limits data and updates,
then creates smoke CSV files only. It does not change the meaning of the full
settings. The supplied original train JSON and raw image hashes must match the
frozen U-Net prediction source before any model is trained.
''')
code('''
MODE='full'
MAX_HOURS=3.4
CFG=refiner_config()
CFG.update(mode=MODE,total_hours=MAX_HOURS,session_started=SESSION_STARTED,gpu_ids=[0,1])
if not torch.cuda.is_available() or max(CFG['gpu_ids'])>=torch.cuda.device_count():
    raise RuntimeError('Select Kaggle GPU T4 x2.')
print('GPUs:',[(i,torch.cuda.get_device_name(i)) for i in CFG['gpu_ids']])
CFG,MANIFEST=prepare_refiner(CFG)
RUN=Path(CFG['run'])
BANK=prepare_sam_bank(CFG,MANIFEST)
print('Bank records:',BANK['count'],'| Hours remaining:',round(sam_remaining(CFG)/3600,2))
''')
md('''
## Cache frozen features; train the SAM mask decoder

Each GPU encodes a disjoint crop shard. Four total loader workers are used.
The encoder runs in float32; finite checks precede and follow float16 storage conversion. A small residual
feature adapter and the mask decoder are trained; the encoder remains fixed. Only completed
records enter training. GT-view variation and dark backgrounds create training
prompts; validation crops never enter gradient updates. The two-GPU DDP decoder
uses foreground BCE, per-crop Dice, a continuity surrogate and learned expected
credited-IoU quality. Best/last checkpoints, losses and GPU resources are saved.
''')
code('''
TRAIN_INDICES,VALID_INDICES=encode_sam_bank(CFG)
CFG=train_refiner(CFG)
print('Encoded train/validation records:',len(TRAIN_INDICES),len(VALID_INDICES))
display(b.load_json(RUN/'inference_plan.json'))
''')
code('''
import pandas as pd
import matplotlib.pyplot as plt
history=pd.read_json(RUN/'decoder_history.jsonl',lines=True)
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].plot(history.epoch,history.training_loss,label='Train')
axes[0].plot(history.epoch,history.validation_loss,'o-',label='Calibration crop loss')
axes[0].legend();axes[0].set_title('Decoder losses (not official PQ)')
axes[1].plot(history.epoch,history.peak_vram_gib);axes[1].set_title('Peak decoder VRAM, GiB')
for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(RUN/'decoder_training.png',dpi=130);plt.show()
''')
md('''
## Calibrate actual proposals, then freeze the selection

The existing U-Net masks propose crops. SAM receives either a point plus a coarse
mask, or a box plus that mask. Refined masks return to the original pixel grid.
Calibration compares refinement and fusion using date-crossfitted quality scores.
The original calibrated pipeline is retained unless the new selection score
exceeds its previous score by the configured minimum gain. This is a selection
criterion, not independent proof of an improvement.
''')
code('''
CALIBRATION=predict_refiner(CFG,'calibration')
RECIPE=calibrate_refiner(CFG,CALIBRATION)
display(pd.DataFrame(b.load_json(RUN/'calibration_grid.json')).head(10))
REPORT_FOLDER=predict_refiner(CFG,'report')
REPORT=evaluate_refiner(CFG,REPORT_FOLDER,RECIPE)
display(pd.DataFrame([REPORT['baseline'],REPORT['selected']],index=['Original baseline','Frozen selected pipeline']))
''')
md('''
## Predict the test set and audit both CSVs

Both selected and original-baseline CSVs are saved. Every exported mask is native
2048×2048, nonempty, pixel-disjoint and checked for exact COCO RLE round trips.
All test sources are accounted for, including images with no predictions. There
are no test labels, manual test edits, synthetic placeholder masks or automatic
leaderboard submissions in this notebook.
''')
code('''
TEST_FOLDER=predict_refiner(CFG,'test')
FILES=export_refiner(CFG,TEST_FOLDER,RECIPE)
from IPython.display import FileLink,display
for name,path in FILES.items():
    print(name);display(FileLink(str(path)))
print('Total elapsed hours:',round((time.time()-SESSION_STARTED)/3600,3))
print('Held-out selected PQ:',REPORT['selected']['PQ'],'| Original baseline:',REPORT['baseline']['PQ'])
print('Public score requires an actual competition submission.')
''')
md('''
## Sources and reproducibility

Generic SAM model/code: [Meta Segment Anything](https://github.com/facebookresearch/segment-anything),
Apache-2.0. The aligned batch wrapper preserves the original pretrained
transformer/heads and changes the image/prompt batching arrangement. Shared I/O
and strict IoU > 0.5 counting follow the provided organizer evaluation notebook.
External solar labels and other participants' predictions are not used.

Keep the run folder with the CSVs: it contains input/source hashes, group records,
training-only indices, frozen-feature cache metadata, encoder throughput, DDP
resources, losses, decoder checkpoints, quality crossfitting audit, all calibration
trials, report comparisons, selection, RLE audits and measured session timing.
The four-hour previous U-Net run is a separate prerequisite, not hidden inside
the decoder-training timing. The actual combined compute ledger must also account
for any smoke/pilot sessions when enforcing an overall ten-hour budget.
''')
nb=nbf.v4.new_notebook(cells=cells)
nb.metadata.update(kernelspec=dict(display_name="Python 3",language="python",name="python3"),
    language_info=dict(name="python",version="3.13"),kaggle=dict(accelerator="nvidiaTeslaT4",
    isGpuEnabled=True,isInternetEnabled=True,language="python",sourceType="notebook",
    dataSources=[dict(sourceType="competition",sourceId=143678)]))
nbf.validate(nb)
target=HERE/"solar-filament-sam-crop-refiner-v4.ipynb"
nbf.write(nb,target)
print(f"Built {target.name}: {len(cells)} cells, {len(combined):,} embedded source characters")
