import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import json
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import solar_unet_full_image_runtime as runtime


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    root = ROOT
    data_root = root / 'MAGFilo_1.0_Kaggle_2026'
    annotations = json.loads((data_root / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    unique = []
    seen = set()
    for info in annotations['images']:
        if info['file_name'] not in seen:
            unique.append(info)
            seen.add(info['file_name'])
        if len(unique) == 3:
            break
    with tempfile.TemporaryDirectory(prefix='solar-full-image-smoke-') as temporary:
        work = Path(temporary)
        records = []
        for index, info in enumerate(unique):
            mask = np.zeros((2048, 2048), dtype=np.uint8)
            for annotation in annotations['annotations']:
                if annotation['image_id'] == info['id']:
                    for polygon in annotation['segmentation']:
                        points = np.round(np.asarray(polygon, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
                        if len(points) >= 3:
                            cv2.fillPoly(mask, [points], 255)
            mask_path = work / f'{index}.png'
            assert cv2.imwrite(str(mask_path), mask)
            records.append(dict(image_path=str((data_root/'train'/'train_images'/info['file_name']).resolve()),
                                mask_path=str(mask_path), image_id=info['id'], file_name=info['file_name']))
        config = dict(train_records=records[:2], val_records=records[2:],
                      image_size=2048, base_channels=16, depth=7, max_channels=256,
                      checkpoint_blocks=True, per_gpu_batch_size=1,
                      memory_fraction=.85, gradient_accumulation_steps=4,
                      seed=10, epochs=1, patience=1, learning_rate=3e-4, weight_decay=1e-4,
                      amp=True, workers=4, prefetch_factor=1, channels_last=True,
                      cudnn_benchmark=False, log_every=1, probe_warmup_steps=1,
                      probe_measure_steps=1, checkpoint_path=str(work/'best.pth'),
                      history_path=str(work/'history.csv'), runtime_report_path=str(work/'runtime_report.json'))
        started = time.perf_counter()
        runtime.train(config)
        checkpoint = torch.load(config['checkpoint_path'], weights_only=True, map_location='cpu')
        model = runtime.FullImageUNet(**checkpoint['model_config']).to('cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(memory_format=torch.channels_last)
        image = cv2.imread(records[2]['image_path'], cv2.IMREAD_GRAYSCALE)
        probabilities = runtime.predict_full_image(model, image, 'cuda')
        assert probabilities.shape == (2048, 2048) and np.isfinite(probabilities).all()
        report = json.loads((work/'runtime_report.json').read_text())
        summary = dict(device=torch.cuda.get_device_name(0), native_image_size=2048,
                       base_channels=16, depth=7, max_channels=256,
                       training_images=2, validation_images=1, loader_workers=4,
                       epochs=1, probe_reports=report['probe_reports'],
                       epoch_memory=report['epoch_memory'],
                       single_forward_output_shape=list(probabilities.shape),
                       elapsed_seconds=time.perf_counter()-started,
                       limitations='Smoke test only: no quality claim, no dual-GPU CUDA measurement.')
        target = root/'tests'/'full_image_gpu_smoke_result.json'
        target.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
