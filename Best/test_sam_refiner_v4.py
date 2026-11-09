"""SAM alignment, leakage, prompt geometry, losses and export ownership checks."""
import ast
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pytest
import torch
from segment_anything.modeling import MaskDecoder,PromptEncoder,TwoWayTransformer

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import sam_refiner_v4_runtime as r


def decoder_and_prompt():
    torch.manual_seed(17)
    decoder=MaskDecoder(transformer_dim=256,transformer=TwoWayTransformer(depth=2,embedding_dim=256,mlp_dim=2048,num_heads=8),
        num_multimask_outputs=3,iou_head_depth=3,iou_head_hidden_dim=256)
    prompt=PromptEncoder(embed_dim=256,image_embedding_size=(64,64),input_image_size=(1024,1024),mask_in_chans=16)
    prompt.eval().requires_grad_(False)
    return decoder,prompt


def test_aligned_decoder_matches_official_single_image_calls():
    decoder,prompt=decoder_and_prompt();model=r.AlignedSAMDecoder(decoder).eval()
    image=torch.randn(2,256,64,64)
    with torch.no_grad():
        sparse,dense=prompt(points=None,boxes=torch.tensor([[10.,20.,600.,700.],[100.,50.,400.,900.]]),masks=torch.randn(2,1,256,256))
        position=prompt.get_dense_pe();mask,quality=model(image,sparse,dense,position)
        for i in range(2):
            expected,score=decoder(image_embeddings=image[i:i+1],image_pe=position,
                sparse_prompt_embeddings=sparse[i:i+1],dense_prompt_embeddings=dense[i:i+1],multimask_output=False)
            assert torch.allclose(mask[i:i+1],expected,atol=1e-5,rtol=1e-5)
            assert torch.allclose(quality[i:i+1],score[:,0],atol=1e-5,rtol=1e-5)


def test_sam_decoder_loss_backward_all_active_parameters_finite():
    decoder,prompt=decoder_and_prompt();model=r.AlignedSAMDecoder(decoder).train()
    image=torch.randn(2,256,64,64)
    with torch.no_grad():
        sparse,dense=prompt(points=None,boxes=torch.tensor([[10.,20.,600.,700.],[100.,50.,400.,900.]]),masks=torch.randn(2,1,256,256))
    mask,quality=model(image,sparse,dense,prompt.get_dense_pe())
    target=torch.zeros(2,1,256,256,dtype=torch.uint8);target[0,:,80:86,30:200]=255
    views=target.expand(-1,2,-1,-1).clone()
    batch=dict(target=target,views=views,nviews=torch.tensor([2,2]))
    loss=r.refiner_loss(mask,quality,batch,torch.device('cpu'));loss.backward()
    assert torch.isfinite(loss) and loss.dtype==torch.float32
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert all(not p.requires_grad for p in prompt.parameters())
    assert all(not p.requires_grad for head in decoder.output_hypernetworks_mlps[1:] for p in head.parameters())


def test_embedding_quantization_is_bounded_and_nonfinite_rejected():
    image=torch.randn(2,256,64,64)
    value,scale=r.quantize_embedding(image)
    restored=value.float()*scale[:,:,None,None]
    assert value.dtype==torch.int8
    assert (restored-image).abs().max() <= scale.max()/2+1e-5
    image[0,0,0,0]=float('nan')
    with pytest.raises(FloatingPointError):r.quantize_embedding(image)


def test_crop_orientation_is_identical_for_image_and_annotation_views():
    image=np.arange(64).reshape(8,8)
    views=np.stack([image,image+100])
    record=dict(rotation=1,flip=True)
    transformed=r.crop_transform(views,record)
    assert np.array_equal(transformed[0],r.crop_transform(image,record))
    assert np.array_equal(transformed[1],transformed[0]+100)


def test_half_storage_preserves_features_and_returns_float_compute_scale():
    features=torch.randn(2,256,64,64)
    stored,scale=r.stored_embedding(features,'float16')
    restored=stored.float()*scale[:,:,None,None]
    assert stored.dtype==torch.float16 and scale.dtype==torch.float32
    assert torch.all(scale==1) and torch.isfinite(restored).all()
    assert (features-restored).abs().max()<.003


def test_relative_weak_seed_preserves_extent_in_prompt_box():
    seed=np.zeros((256,256),np.float32);seed[100:105,20:220]=.20
    mask,point,box=r.prompt_tensors(seed)
    assert box[0]<80 and box[2]>880  # weak masks must not collapse to a single point
    assert 80<=point[0,0]<=880 and 400<=point[0,1]<=420
    assert mask.shape==(1,256,256) and np.isfinite(mask).all()


def test_square_roi_stays_on_original_grid():
    cfg=r.refiner_config()
    for box in ([0,2,50,60],[2000,2000,2048,2048],[900,800,1300,1000]):
        x0,y0,x1,y1=r.square_roi(box,cfg)
        assert 0<=x0<x1<=2048 and 0<=y0<y1<=2048 and x1-x0==y1-y0
        assert x0<=box[0] and y0<=box[1] and x1>=box[2] and y1>=box[3]


def test_native_seed_mapping_respects_roi_offset():
    mask=np.zeros((10,20),bool);mask[2:7,3:15]=True
    candidate=dict(box=[500,600,520,610],local_rle=r.b.encode(mask))
    p=np.ones((2048,2048),np.float32)*.7
    placed=r.put_seed_in_roi(candidate,[480,580,544,644],p)
    assert placed.shape==(64,64)
    assert float(placed.sum())==pytest.approx(mask.sum()*.7,rel=1e-6)
    assert np.all(placed[22:27,23:35]>.69)


def test_fusion_trims_overlap_and_preserves_prior_recipe():
    def candidate(box,score):
        mask=np.ones((10,10),bool)
        return dict(box=box,local_rle=r.b.encode(mask),rle=r.b.full_rle(box,mask),area=100,features=[score])
    first=candidate([100,100,110,110],.8);second=candidate([106,100,116,110],.7)
    class Quality:
        def predict(self,x):return x[:,0]
    sets={'baseline':[first],'refined_0.5':[second]}
    result=r.select_refiner_instances(sets,dict(family='fusion',threshold=.5,cutoff=.1,min_area=10),Quality())
    assert len(result)==2
    masks=[r.b.decode(x) for x in result]
    assert not np.any(masks[0]&masks[1]) and sum(int(x.sum()) for x in masks)==160
    assert r.select_refiner_instances(sets,dict(family='prior'))==[first['rle']]


def test_calibration_uses_out_of_fold_prior_scores(tmp_path,monkeypatch):
    (tmp_path/'selected_pipeline.json').write_text(json.dumps(dict(recipe=dict(set='s',quality=True))))
    folder=tmp_path/'predictions'/'hash'/'calibration'/'candidates_x';folder.mkdir(parents=True)
    sources=[]
    for stem,y in [('A',.1),('B',.9)]:
        (folder/(stem+'.json')).write_text(json.dumps({'s':[dict(features=[0.]*18,target=y)]}))
        sources.append(dict(stem=stem,date=stem))
    class Estimator:
        def fit(self,x,y):self.mean=float(y.mean());return self
        def predict(self,x):return np.full(len(x),self.mean)
    monkeypatch.setattr(r.b,'quality_estimator',lambda seed:Estimator())
    result=r.prior_crossfit_scores(dict(run=str(tmp_path)),tmp_path,
        dict(config=dict(seed=3,quality_folds=2),splits=dict(calibration=sources)))
    assert result['A'][0]==pytest.approx(.9) and result['B'][0]==pytest.approx(.1)


def test_prior_filter_does_not_call_fitted_regressor_on_calibration(tmp_path):
    path=tmp_path/'c.json'
    path.write_text(json.dumps({'s':[dict(features=[0.]*18,area=100,rle={'counts':'abc'})]}))
    class Forbidden:
        def predict(self,x):raise AssertionError('Full-fit quality must not score calibration')
    manifest=dict(prior_calibration=dict(recipe=dict(set='s',quality=True,cutoff=.25,min_area=40)))
    source=dict(prior_candidates=str(path),prior_oof=[.8])
    assert len(r.prior_instances(source,manifest,Forbidden()))==1


def test_notebook_is_standalone_and_worker_matches_ast():
    notebook=nbformat.read(HERE/'solar-filament-sam-crop-refiner-v4.ipynb',as_version=4)
    nbformat.validate(notebook);sections=[]
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        source=cell.source
        if source.startswith('%%sam_section'):
            source=source.split('\n',1)[1];sections.append(source)
        compile(source,'notebook-cell','exec')
        assert cell.execution_count is None and not cell.outputs
    worker='\n\n'.join(sections)+"\nif __name__ == '__main__':\n    refiner_main()\n"
    assert ast.dump(ast.parse(worker))==ast.dump(ast.parse((HERE/'sam_refiner_v4_worker.py').read_text(encoding='utf8')))
    assert 'import pq50_v3_runtime' not in worker
    assert notebook.metadata.kaggle.isGpuEnabled is True


def test_previous_quality_pickle_loads_without_notebook_import_state():
    paths=list((HERE/'analysis_2026_11_08'/'kaggle_pilot_output').glob('*/quality_model.joblib'))
    if not paths:pytest.skip('User run artifact is not present')
    model=r.load_quality(paths[0])
    predicted=model.predict(np.zeros((2,18),np.float32))
    assert predicted.shape==(2,) and np.isfinite(predicted).all()
