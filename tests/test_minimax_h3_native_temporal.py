import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import minimax_h3_native_temporal as node

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
CORE=load('native_temporal_comfy_fixture',ROOT/'tests/fixtures/comfy_h3_layout_8cfe5e1.py')

def example(vt=7):
 refs=[dict(kind='image',latent_h=4,latent_w=8),dict(kind='video_audio',latent_h=8,latent_w=4,latent_t=vt,ref_audio_t=3)]
 l=CORE.PackedLayout(3,7,8,8,2,refs=refs)
 return l,refs

def test_layout_matches_actual_training_coordinates():
 l,refs=example();before=l.position_ids.clone()
 aligned,src,tgt=node.temporal_layout(l,refs,None)
 golden=json.loads((ROOT/'tests/fixtures/h3_native_temporal_training_positions.json').read_text())
 torch.testing.assert_close(aligned.position_ids,torch.tensor(golden['position_ids'],dtype=torch.float64),rtol=0,atol=1e-12)
 assert src.tolist()==golden['source']
 assert torch.equal(l.position_ids,before)
 assert torch.equal(aligned.position_ids[~src],before[~src])
 assert torch.equal(aligned.position_ids[src,1:],before[src,1:])
 assert not (src&tgt).any()

def test_wrong_video_index_and_temporal_mismatch_rejected():
 l,r=example()
 with pytest.raises(ValueError,match='Video 2'):node.temporal_layout(l,r,None,2)
 l,r=example(6)
 with pytest.raises(ValueError,match='identical temporal'):node.temporal_layout(l,r,None)

def test_add_guide_history_keyframes_keep_reference_alignment():
 # The previous shot's last frames as an Add Guide history (cond rows after the text) must not shift the reference.
 l,refs=example();plain,src0,tgt0=node.temporal_layout(l,refs,None)
 kfs=[dict(resolved_frame_index=0,latent=torch.zeros(1,4,2,8,8))]
 k=CORE.PackedLayout(3,7,8,8,2,keyframes=kfs,refs=refs)
 aligned,src,tgt=node.temporal_layout(k,refs,kfs)
 cond=[s for s in k.segments if s[2]=='cond'];assert cond
 a,b,_=cond[0];n=b-a
 assert src.sum()==src0.sum() and tgt.sum()==tgt0.sum()
 assert not src[a:b].any() and not tgt[a:b].any()
 # Same rows as without history, shifted by the cond rows; same aligned positions for the reference and the target.
 assert torch.equal(src[b:],src0[a:]) and torch.equal(tgt[b:],tgt0[a:])
 torch.testing.assert_close(aligned.position_ids[src],plain.position_ids[src0],rtol=0,atol=1e-12)
 torch.testing.assert_close(aligned.position_ids[tgt],plain.position_ids[tgt0],rtol=0,atol=1e-12)
 # History rows sit on the target's first frames and are left untouched.
 torch.testing.assert_close(aligned.position_ids[a:b],k.position_ids[a:b],rtol=0,atol=0)
 assert aligned.position_ids[a,0]==aligned.position_ids[tgt][0,0]

def context():
 l,r=example();a,s,t=node.temporal_layout(l,r,None)
 return node.core.TemporalContext(a.position_ids[:,0].float()[None]/40,s[None],t[None],torch.ones_like(s[None]),2,.2,.25,6)

def test_attention_adapter_matches_dense_score_policy_and_shapes():
 ctx=context();torch.manual_seed(12)
 q,k,v=[torch.randn(1,2,ctx.times.shape[1],8) for _ in range(3)]
 def forbidden(*args,**kwargs):raise AssertionError('wrong backend')
 out=node.attention_override(ctx)(forbidden,q,k,v,2,skip_reshape=True)
 expected=node.core.dense_reference(q,k,v,ctx).transpose(1,2).flatten(2)
 torch.testing.assert_close(out,expected)
 raw=[x.transpose(1,2).flatten(2) for x in (q,k,v)]
 torch.testing.assert_close(node.attention_override(ctx)(forbidden,*raw,2),expected)

def test_token_refiner_uses_existing_attention():
 ctx=context();q=torch.randn(1,2,3,8)
 def original(*args,**kwargs):return 'original'
 assert node.attention_override(ctx)(original,q,q,q,2,skip_reshape=True)=='original'

class Model:
 def __init__(self):self.model_options={}
 def clone(self):m=Model();m.model_options=copy.deepcopy(self.model_options);return m
 def get_model_object(self,key):return type('MiniMaxH3Model',(),{})()
 def set_model_unet_function_wrapper(self,w):self.model_options['model_function_wrapper']=w

def test_per_model_wrapper_clones_payload_and_applies_attention():
 m=Model();patched=node.patch_temporal_model(m);assert not m.model_options
 l,r=example();payload=dict(layout=l,refs=r)
 def fn(x,t,**c):
  assert c['minimax_payload']['layout'] is not l
  assert 'optimized_attention_override' in c['transformer_options']
  return 'ok'
 args=dict(input=torch.zeros(1),timestep=torch.ones(1),c=dict(minimax_payload=payload))
 assert patched.model_options['model_function_wrapper'](fn,args)=='ok'
 assert payload['layout'] is l and 'transformer_options' not in args['c']
 with pytest.raises(ValueError,match='already installed'):node.patch_temporal_model(patched)

def test_incompatible_attention_rejected():
 m=Model();m.model_options={'transformer_options':{'optimized_attention_override':object()}}
 with pytest.raises(ValueError,match='VSA'):node.patch_temporal_model(m)
 with pytest.raises(ValueError,match='MODEL input'):node.patch_temporal_model(None)


def test_second_video_selection_preserves_first_video_and_images():
 refs=[dict(kind='image',latent_h=4,latent_w=8),
       dict(kind='video',latent_h=4,latent_w=4,latent_t=3,ref_audio_t=0),
       dict(kind='video',latent_h=8,latent_w=4,latent_t=7,ref_audio_t=0)]
 native=CORE.PackedLayout(3,7,8,8,2,refs=refs)
 aligned,source,target=node.temporal_layout(native,refs,None,2)
 assert int(source.sum())==56
 assert torch.equal(aligned.position_ids[~source],native.position_ids[~source])
 assert not source[3:23].any()

@pytest.mark.parametrize('install_after_temporal',[False,True])
@pytest.mark.parametrize('dense_name,expected_backend',[('comfy_kitchen_int8',2),('pytorch',0)])
def test_model_attention_backend_is_honored_without_manual_selection(monkeypatch,install_after_temporal,dense_name,expected_backend):
 from types import ModuleType
 mod=ModuleType('comfy.ldm.modules.attention')
 def dense(*args,**kwargs):return 'kitchen'
 mod.get_attention_function=lambda name,default=None:dense if name==dense_name else default
 monkeypatch.setitem(sys.modules,'comfy.ldm.modules.attention',mod)
 def setter(optimized_attention):
  def optimized_attention_override(_, *args, **kwargs):return optimized_attention(*args,**kwargs)
  optimized_attention_override.__module__='comfy.model_patcher'
  optimized_attention_override.__qualname__='ModelPatcher.set_model_optimized_attention.<locals>.optimized_attention_override'
  return optimized_attention_override
 incoming=setter(dense)
 m=Model()
 if not install_after_temporal:m.model_options={'transformer_options':{'optimized_attention_override':incoming}}
 patched=node.patch_temporal_model(m)
 selected=[]
 def capture(ctx,previous=None,backend=None,**kwargs):
  selected.append((backend,previous));return 'temporal attention'
 monkeypatch.setattr(node,'attention_override',capture)
 l,r=example()
 args=dict(input=torch.zeros(1),timestep=torch.ones(1),c=dict(minimax_payload=dict(layout=l,refs=r),transformer_options={'optimized_attention_override':incoming}))
 def fn(x,t,**c):
  assert c['transformer_options']['optimized_attention_override']=='temporal attention'
  return 'ok'
 assert patched.model_options['model_function_wrapper'](fn,args)=='ok'
 assert selected==[(node.BACKENDS[expected_backend],incoming)]

@pytest.mark.parametrize('hidden_by_later_dense_node',[False,True])
def test_sol_node_owned_producer_is_bypassed_but_temporal_override_remains(monkeypatch,hidden_by_later_dense_node):
 Patch=type('SparseAttnPatch',(),{'__module__':'comfy_extras.nodes_sparse_attention'})
 patch=Patch();patch.vsa=False;patch.topk_ratio=0.
 previous=None
 def sparse_override(*args):return patch,previous,args
 sparse_override.__module__='comfy_extras.nodes_sparse_attention'
 sparse_override.__qualname__='make_attention_override.<locals>.override'
 def block_patch(*args):return patch,args
 block_patch.__module__=sparse_override.__module__
 block_patch.__qualname__='make_h3_block_patch.<locals>.block_patch'
 m=Model();m.model_options={'transformer_options':{'optimized_attention_override':None if hidden_by_later_dense_node else sparse_override,'patches_replace':{'dit':{'block':block_patch}}}}
 patched=node.patch_temporal_model(m)
 selected=[]
 def capture(ctx,previous=None,backend=None,sparse_patch=None,**kw):
  selected.append((backend,sparse_patch));return 'temporal sol hook'
 monkeypatch.setattr(node,'attention_override',capture)
 l,r=example();opts=patched.model_options['transformer_options']
 args=dict(input=torch.zeros(1),timestep=torch.ones(1),c=dict(minimax_payload=dict(layout=l,refs=r),transformer_options=opts))
 def fn(x,t,**c):
  assert c['transformer_options']['patches_replace']['dit']=={}
  assert c['transformer_options']['optimized_attention_override']=='temporal sol hook'
  return 'ok'
 assert patched.model_options['model_function_wrapper'](fn,args)=='ok'
 assert selected==[(node.SOL_BACKEND,patch)]
 assert m.model_options['transformer_options']['patches_replace']['dit']


def test_sol_dense_window_preserves_temporal_policy(monkeypatch,capsys):
 l,r=example();aligned,src,tgt=node.temporal_layout(l,r,None)
 ctx=node.core.TemporalContext(aligned.position_ids[:,0].float()[None]/40,src[None],tgt[None],torch.ones(1,l.seq_len,dtype=torch.bool),2.,.2,.25,6.)
 plan=node.fast.make_plan(aligned.position_ids,src,tgt,2.,.2,.25,6.)
 patch=SimpleNamespace(dense_reason=lambda *a:'outside sigma window',log_once=lambda *a:None)
 q,k,v=[torch.randn(1,2,l.seq_len,128) for _ in range(3)]
 monkeypatch.setattr(node.fast,'attention',lambda q,k,v,p:node.core.dense_reference(q,k,v,ctx))
 expected=node.core.dense_reference(q,k,v,ctx)
 out=node.attention_override(ctx,backend=node.SOL_BACKEND,plan=plan,sparse_patch=patch,dense_backend=node.BACKENDS[0],log_first=True)(None,q,k,v,2,skip_reshape=True,skip_output_reshape=True)
 torch.testing.assert_close(out,expected)
 assert 'dense fallback: outside sigma window' in capsys.readouterr().out


def test_sol_sparse_window_runs_temporal_kernel_and_logs_activation(monkeypatch,capsys):
 from types import ModuleType
 ck=ModuleType('comfy_kitchen');ck.sol_attn_is_available=lambda device:True
 monkeypatch.setitem(sys.modules,'comfy_kitchen',ck)
 l,r=example();aligned,src,tgt=node.temporal_layout(l,r,None)
 ctx=node.core.TemporalContext(aligned.position_ids[:,0].float()[None]/40,src[None],tgt[None],torch.ones(1,l.seq_len,dtype=torch.bool),2.,.2,.25,6.)
 plan=node.fast.make_plan(aligned.position_ids,src,tgt,2.,.2,.25,6.)
 patch=SimpleNamespace(dense_reason=lambda *a:None,sinks=lambda *a:((0,0),(0,0)),tau=1.3,extra_tokens=256)
 actual_sol=node.sol.attention
 calls=[]
 def oracle(q,k,v,key_bias=None,**kw):
  calls.append(key_bias)
  mask=None if key_bias is None else key_bias.view(1,1,1,-1)
  return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask).transpose(1,2)
 monkeypatch.setattr(node.sol,'attention',lambda *a,**kw:actual_sol(*a,**kw,kernel=oracle))
 q,k,v=[torch.randn(1,2,l.seq_len,128) for _ in range(3)]
 expected=node.core.dense_reference(q,k,v,ctx)
 out=node.attention_override(ctx,backend=node.SOL_BACKEND,plan=plan,sparse_patch=patch,dense_backend=node.BACKENDS[0])(None,q,k,v,2,skip_reshape=True,skip_output_reshape=True)
 torch.testing.assert_close(out,expected,atol=1e-6,rtol=1e-5)
 assert len(calls)==plan.frames+1
 assert 'sol-attn ACTIVE' in capsys.readouterr().out


def test_node_prepares_selected_kitchen_before_sampling_but_not_flash(monkeypatch):
 calls=[]
 monkeypatch.setattr(node.sol.native_kernel,'prepare_model',lambda model:calls.append(model))
 m=Model();m.load_device=torch.device('cuda:0')
 node.patch_temporal_model(m,backend=node.BACKENDS[0])
 assert calls==[]
 node.patch_temporal_model(m,backend=node.BACKENDS[2])
 assert calls==[m]


def test_structure_free_payload_drops_only_the_motion_video():
 l,r=example()
 refs=[dict(r[0],latent=torch.ones(1)),dict(r[1],latent=torch.full((1,),2.),audio_latent=torch.full((1,),3.))]
 free=node.structure_free_payload(dict(layout=l,refs=refs,seed=7),l,1)
 assert [x['kind'] for x in free['refs']]==['image'] and free['seed']==7
 assert [float(x) for x in free['cond_video_latents']]==[1.] and free['cond_audio_latents']==[]
 assert free['layout'].signature==l.signature and free['layout'].seq_len==l.seq_len-2*3-7*4*2
 assert [k for _,_,k in free['layout'].segments]==['text','ref_img','audio','video']


def test_structure_guidance_extra_pass_without_video_and_sigma_cutoff():
 m=Model();patched=node.patch_temporal_model(m,structure_guidance=1.5,structure_sigma_min=.5)
 l,r=example();calls=[]
 def fn(x,t,**c):
  video='optimized_attention_override' in c.get('transformer_options',{})
  assert video==any(ref['kind']=='video_audio' for ref in c['minimax_payload']['refs'])
  calls.append(video);return torch.full((1,),3. if video else 1.)
 wrapper=patched.model_options['model_function_wrapper']
 out=wrapper(fn,dict(input=torch.zeros(1),timestep=torch.full((1,),.8),c=dict(minimax_payload=dict(layout=l,refs=r))))
 assert calls==[True,False] and float(out)==4.  # 3 + 0.5 * (3 - 1)
 calls.clear()
 out=wrapper(fn,dict(input=torch.zeros(1),timestep=torch.full((1,),.2),c=dict(minimax_payload=dict(layout=l,refs=r))))
 assert calls==[True] and float(out)==3.
 with pytest.raises(ValueError,match='structure_guidance'):node.patch_temporal_model(Model(),structure_guidance=.5)
