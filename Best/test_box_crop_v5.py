import ast
import copy
import time
from pathlib import Path

import cv2
import nbformat
import numpy as np
import pytest
import torch

import box_crop_v5_runtime as r


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)


def test_input_hint_is_a_rectangle_and_has_no_mask_boundary():
    gray=np.full((2048,2048),127,np.uint8)
    image=r.box_input(gray,[100,110,160,150],[80,80,200,200],128)
    hint=image[2]*.225+.406
    assert np.all(np.isclose(hint,0)|np.isclose(hint,1))
    y,x=np.where(hint>.5)
    assert len(x)==(x.max()-x.min()+1)*(y.max()-y.min()+1)
    assert image.shape==(3,128,128)
    assert np.isfinite(image).all()


def test_crop_targets_align_and_preserve_annotation_identity(tmp_path):
    image=np.full((2048,2048,3),130,np.uint8)
    a=np.zeros((2048,2048),bool);a[950:965,960:1050]=1;a[965:1000,960:975]=1
    z=np.zeros_like(a);z[955:975,960:1040]=1
    image[a,0]=30
    path=tmp_path/'image.png';assert cv2.imwrite(str(path),image)
    row=dict(image=str(path),roi=[900,900,1100,1100],bounds=[960,950,1050,1000],
             rles=[r.b.encode(a),r.b.encode(z),None],nviews=3,positive=True)
    cfg={**r.crop_config(),"crop_size":128}
    dataset=r.BoxCropDataset([row],cfg,True)
    result=dataset[0]
    assert result['nviews']==3
    assert torch.equal(result['views'][2],torch.zeros_like(result['views'][2]))
    assert any(torch.equal(result['target'][0],v) for v in result['views'][:2])
    hint=result['image'][2]*.225+.406
    assert torch.all(torch.isclose(hint,torch.tensor(0.))|torch.isclose(hint,torch.tensor(1.)))
    assert result['target'].max()<=1 and result['target'].min()>=0
    assert not torch.equal(result['views'][0],result['views'][1])


def test_piece_pool_compares_fragments_with_their_union():
    p=np.zeros((80,90),np.float32);p[10:20,10:30]=.9;p[40:45,50:70]=.9
    result=r.supported_pieces(p,p,.5)
    areas=sorted(int(mask.sum()) for box,mask in result)
    assert areas==[100,200,300]


@pytest.mark.parametrize('encoder',['resnet34','tu-tf_efficientnetv2_s'])
def test_production_model_all_active_gradients_are_finite(encoder):
    model=r.BoxCropUNet(dict(encoder=encoder),False).train()
    assert all(not module.training for module in model.core.encoder.modules()
               if isinstance(module,torch.nn.modules.batchnorm._BatchNorm))
    x=torch.randn(2,3,128,128)
    target=torch.zeros(2,1,128,128);target[:,:,55:62,30:100]=1
    batch=dict(target=target,views=target.repeat(1,3,1,1),nviews=torch.tensor([3,3]))
    outputs=model(x)
    loss=r.crop_loss(*outputs,batch,torch.device('cpu'))
    assert torch.isfinite(loss)
    loss.backward()
    missing=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_quality_target_accounts_for_missing_annotator_view():
    target=torch.zeros(1,1,32,32);target[:,:,10:15,8:25]=1
    logits=(target*2-1)*20
    views=torch.cat([target,target*0,target*0],dim=1)
    batch=dict(target=target,views=views,nviews=torch.tensor([3]))
    low=r.crop_loss(logits,torch.tensor([1/3]),batch,torch.device('cpu'))
    high=r.crop_loss(logits,torch.tensor([1.]),batch,torch.device('cpu'))
    assert low<high


def test_fallback_exports_identical_native_masks_without_neural_inference(tmp_path,monkeypatch):
    mask=np.zeros((2048,2048),bool);mask[500:510,800:830]=True
    candidate=dict(rle=r.b.encode(mask))
    cfg={**r.crop_config(),"mode":"smoke","run":str(tmp_path),"output_root":str(tmp_path),
         "session_started":time.time()}
    manifest=dict(splits=dict(test=[dict(stem="example",name="example.jpeg")]))
    r.b.save_json(tmp_path/'manifest.json',manifest)
    monkeypatch.setattr(r.s,'prior_quality',lambda cfg:None)
    monkeypatch.setattr(r.s,'prior_instances',lambda source,manifest,quality:[copy.deepcopy(candidate)])
    monkeypatch.setattr(r.s,'load_quality',lambda path:None)
    folder=r.baseline_candidates(cfg,'test')
    files=r.export_crops(cfg,folder,dict(family='prior'))
    assert files['selected'].read_bytes()==files['baseline'].read_bytes()
    audit=r.b.load_json(tmp_path/'submission_audits.json')
    assert audit['selected']['masks']==1
    assert audit['selected']['roundtrip'] and audit['selected']['nonoverlap']
    assert audit['selected']['native_shape']==[2048,2048]


def test_worker_matches_all_runtime_functions_and_notebook_capture():
    root=Path(__file__).resolve().parent
    runtime=ast.parse((root/'box_crop_v5_runtime.py').read_text(encoding='utf8'))
    worker=ast.parse((root/'box_crop_v5_worker.py').read_text(encoding='utf8'))
    nodes={node.name:ast.dump(node,include_attributes=False) for node in worker.body
           if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    for node in runtime.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            assert nodes[node.name]==ast.dump(node,include_attributes=False)
    notebook=nbformat.read(root/'solar-filament-box-crop-v5.ipynb',as_version=4)
    chunks=[cell.source.split('\n',1)[1] for cell in notebook.cells
            if cell.cell_type=='code' and cell.source.startswith('%%crop_section')]
    captured=ast.parse('\n\n'.join(chunks)+"\nif __name__=='__main__':\n    crop_main()\n")
    assert ast.dump(captured,include_attributes=False)==ast.dump(worker,include_attributes=False)
