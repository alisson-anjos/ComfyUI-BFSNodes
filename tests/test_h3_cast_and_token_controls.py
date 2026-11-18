import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bfs_h3_cast_bindings as cast
import minimax_h3_token_weights as tokens


def test_selection_indices_and_coordinates_are_validated():
    valid=dict(source='test',bindings=[dict(picture=2,frame=3,positive=[dict(x=.25,y=.75)],negative=[])])
    assert cast.parse_selection(json.dumps(valid),2,5)==valid
    with pytest.raises(ValueError):cast.parse_selection(json.dumps(valid),1,5)
    valid['bindings'][0]['positive'][0]['x']=float('nan')
    with pytest.raises(ValueError):cast.parse_selection(json.dumps(valid),2,5)


def test_native_sam_tracking_coverage_and_keyframe_order(monkeypatch):
    class Track:
        @classmethod
        def execute(cls,images,**kwargs):return SimpleNamespace(args=(images,))
    class ToMask:
        @classmethod
        def execute(cls,track_data,**kwargs):return SimpleNamespace(args=(track_data[...,0],))
    class Detect:
        @classmethod
        def execute(cls,image,**kwargs):return SimpleNamespace(args=(torch.ones(image.shape[:3]),))
    monkeypatch.setitem(sys.modules,'comfy_extras.nodes_sam3',SimpleNamespace(SAM3_Detect=Detect,SAM3_VideoTrack=Track,SAM3_TrackToMask=ToMask))
    work=torch.arange(302).float()[:,None,None,None].expand(302,2,2,3)
    out=cast.track_sequence(work,None,torch.ones(1,2,2),.5)
    torch.testing.assert_close(out,work[...,0])
    monkeypatch.setattr(cast,'preview_video',lambda *args:dict(filename='preview.mp4'))
    selection=dict(bindings=[dict(picture=1,frame=2,positive=[dict(x=.5,y=.5)],negative=[])])
    frames=torch.arange(5).float()[:,None,None,None].expand(5,4,4,3)
    result=cast.BFSH3SAM31CastMasks().track(frames,None,selection,max_tracking_side=128)
    torch.testing.assert_close(result['result'][0]['bindings'][0]['mask'],frames[...,0])


def test_loop_masks_slice_and_pad_without_changing_picture_binding(monkeypatch):
    seen=[]
    class Binder:
        def apply(self,model,**kwargs):seen.append(kwargs);return (model,)
    monkeypatch.setattr(cast,'BFSMiniMaxH3ReferenceBinding',Binder)
    masks=dict(source_frames=8,bindings=[dict(picture=2,mask=torch.arange(8).float()[:,None,None])])
    state=dict(source_frames=8,index=1,ranges=[dict(start=0,end=5),dict(start=5,end=8)])
    model=object();assert cast.BFSH3ApplyCastBindings().apply(model,masks,5,loop_state=state)[0] is model
    assert seen[0]['picture_index']==2
    assert seen[0]['target_mask'][:,0,0].tolist()==[5,6,7,7,7]


def traced():
    decoder=SimpleNamespace(decode=lambda ids,**kwargs:{10:'blue',11:' shirt'}[ids[0]])
    clip=SimpleNamespace(tokenize=lambda prompt:{'qwen3vl_32b':[[(10,1.),(11,1.)]]},tokenizer=SimpleNamespace(qwen3vl_32b=decoder))
    embedding=torch.arange(20).float().reshape(1,5,4)
    meta=dict(minimax_token_tags=torch.tensor([1,0,0,1,1]),minimax_refs=[{'kind':'image'}])
    return tokens.attach_trace([[embedding,meta]],clip,'blue shirt'),embedding,meta


def test_prompt_trace_and_embedding_weights_preserve_vision_and_metadata():
    conditioning,original,meta=traced();trace=conditioning[0][1][tokens.TRACE]
    assert trace['start']==3 and trace['tokens'][0]['label']=='blue'
    weights=json.dumps(dict(prompt_sha256=trace['prompt_sha256'],weights={'1':1.5}))
    output=tokens.BFSH3TokenWeightEditor().edit(conditioning,None,mode='embedding magnitude (experimental)',inspect_only=False,weights_json=weights)['result'][0]
    torch.testing.assert_close(output[0][0][:,:4],original[:,:4])
    torch.testing.assert_close(output[0][0][:,4],original[:,4]*1.5)
    assert output[0][1]['minimax_refs'] is meta['minimax_refs']
    assert torch.equal(conditioning[0][0],original)
    with pytest.raises(ValueError,match='Prompt changed'):tokens.parse_weights('{"prompt_sha256":"old","weights":{}}',trace)


def test_token_attention_weights_clone_model_and_keep_conditioning_unchanged():
    class Model:
        def __init__(self):self.model_options={'bfs_h3_native_temporal_v1':{'version':'v1'}}
        def clone(self):m=Model();m.model_options=dict(self.model_options);return m
    conditioning,_,_=traced();m=Model()
    out=tokens.BFSH3TokenWeightEditor().edit(conditioning,m,inspect_only=False,weights_json='{"weights":{"0":1.25,"1":1.25}}')['result']
    assert out[0] is conditioning and out[1] is not m
    policy=out[1].model_options['transformer_options'][tokens.OPTION]
    assert policy['ranges']==[(3,5,1.25)] and policy['sequence_length']==5
    assert tokens.OPTION not in m.model_options.get('transformer_options',{})


def test_token_attention_weights_work_without_tara_via_direct_override():
    import math
    import torch
    class Model:
        def __init__(self):self.model_options={};self.wrapper=None
        def clone(self):m=Model();m.model_options=dict(self.model_options);m.wrapper=self.wrapper;return m
        def set_model_unet_function_wrapper(self,fn):self.wrapper=fn
    conditioning,_,_=traced();m=Model()
    out=tokens.BFSH3TokenWeightEditor().edit(
        conditioning,m,inspect_only=False,
        weights_json='{"weights":{"0":2.0,"1":0.2}}',
        attention_strength=2.0,
    )['result']
    assert out[0] is conditioning and out[1] is not m and out[1].wrapper is not None
    captured={}
    def fake_model(x,t,**c):
        override=c['transformer_options']['optimized_attention_override']
        q=torch.zeros((1,2,8,4))
        k=torch.zeros((1,2,8,4))
        v=torch.ones((1,2,8,4))
        def attn(q_,k_,v_,heads,mask=None,**kw):
            captured['mask']=mask
            captured['v']=v_
            return v_
        return override(attn,q,k,v,2)
    out[1].wrapper(fake_model,{'input':torch.zeros(1),'timestep':torch.zeros(1),'c':{'transformer_options':{'cond_or_uncond':[0]}}})
    assert math.isclose(float(captured['mask'][0,0,0,3]),2.0*math.log(2.0),rel_tol=1e-5)
    assert math.isclose(float(captured['mask'][0,0,0,4]),2.0*math.log(0.2),rel_tol=1e-5)
    assert math.isclose(float(captured['v'][0,0,3,0]),1.0,rel_tol=1e-5)
    assert math.isclose(float(captured['v'][0,0,4,0]),0.2,rel_tol=1e-5)


def test_token_editor_report_formats():
    conditioning,_,_=traced()
    editor=tokens.BFSH3TokenWeightEditor()
    compact=editor.edit(
        conditioning,None,mode='embedding magnitude (experimental)',inspect_only=False,
        weights_json='{"weights":{"0":0.05,"1":3.0}}',report_format='compact (changes)',
    )['result'][2]
    assert compact=='embed | (blue:0.05) (shirt:3)'
    annotated=editor.edit(
        conditioning,None,mode='embedding magnitude (experimental)',inspect_only=False,
        weights_json='{"weights":{"0":0.05}}',report_format='annotated prompt',
    )['result'][2]
    assert annotated=='(blue:0.05) shirt'
    plain=editor.edit(
        conditioning,None,mode='embedding magnitude (experimental)',inspect_only=False,
        weighted_phrases='blue shirt',report_format='compact (changes)',
    )['result'][2]
    assert plain=='embed | baseline (no overrides)'


def test_new_example_graphs_have_no_cycles_and_keep_original_audio():
    for suffix in ('','_loop'):
        d=json.loads((ROOT/f'examples/h3_tara_cast_sam31{suffix}.api.json').read_text())
        visiting=set();done=set()
        def visit(n):
            assert n not in visiting,'Cycle in example API graph'
            if n in done:return
            visiting.add(n)
            for value in d[n]['inputs'].values():
                if isinstance(value,list) and len(value)==2 and isinstance(value[0],str):visit(value[0])
            visiting.remove(n);done.add(n)
        for n in d:visit(n)
        assert d['62']['inputs']['model']==['61',0]
        assert d['63']['inputs']['cast_masks']==['62',0]
        assert d['15']['inputs']['model']==['64',1]
        if suffix:
            assert d['63']['inputs']['loop_state']==['40',1]
            assert d['22']['inputs']['audio']==['10',2]
        ui=json.loads((ROOT/f'examples/h3_tara_cast_sam31{suffix}.comfy.json').read_text());nodes={n['id']:n for n in ui['nodes']}
        for lid,src,slot,dst,input_slot,kind in ui['links']:
            assert lid in nodes[src]['outputs'][slot]['links']
            assert nodes[dst]['inputs'][input_slot]['link']==lid
