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