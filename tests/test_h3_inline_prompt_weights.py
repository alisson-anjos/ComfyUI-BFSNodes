import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import minimax_h3_token_weights as w
import bfs_h3_native_workflow as workflow


def clip_with_offsets():
    def encode(text,**kwargs):
        return dict(input_ids=[ord(c) for c in text],offset_mapping=[(i,i+1) for i in range(len(text))])
    decoder=SimpleNamespace(tokenizer=encode,decode=lambda ids,**kw:''.join(chr(i) for i in ids))
    return SimpleNamespace(tokenizer=SimpleNamespace(qwen3vl_32b=decoder),tokenize=lambda text:{'qwen3vl_32b':[[(ord(c),1.) for c in text]]})


def traced(prompt,inline=None):
    tensor=torch.ones(1,3+len(prompt),4)
    meta=dict(minimax_token_tags=torch.tensor([1,0,0]+[1]*len(prompt)),minimax_refs=[{'kind':'image'}])
    return w.attach_trace([[tensor,meta]],clip_with_offsets(),prompt,inline_spans=inline),tensor


def test_explicit_syntax_nested_literal_escapes_and_invalid_forces():
    clean,spans=w.parse_prompt_weights('Keep (the (blue:1.2) shirt:1.3) and (ordinary words).')
    assert clean=='Keep the blue shirt and (ordinary words).'
    assert [(clean[s['start']:s['end']],s['gain']) for s in spans]==[('blue',1.2),('the blue shirt',1.3)]
    assert w.parse_prompt_weights(r'Keep \(blue:1.3\) literal')==('Keep (blue:1.3) literal',[])
    assert w.parse_prompt_weights('a (blue:1.3')==('a (blue:1.3',[])
    for text in ('At (00:03.333), move.', '(Shot 1 at 00:03)', 'Ratio (32:64)'):
        assert w.parse_prompt_weights(text)==(text,[])
    for text in ('(red:-2.1)','(red:3.1)','(:1.2)'):
        with pytest.raises(ValueError):w.parse_prompt_weights(text)


def test_repeated_phrase_boundaries_unicode_and_clean_full_prompt():
    cond,_=traced('A man and a woman; a man wearing café red.')
    trace=cond[0][1][w.TRACE]
    gains=w.phrase_weights('(man:1.3)\n(café red:0.8)',trace)
    for token in trace['tokens']:
        i=token['index'];start=token['char_start']
        if 2<=start<5 or 21<=start<24:assert gains[i]==pytest.approx(1.3)
        if 12<=start<17:assert i not in gains # woman is not the word man
    assert any(g==.8 for g in gains.values())
    with pytest.raises(ValueError,match='not present'):w.phrase_weights('(goblin:1.2)',trace)
    clean,spans=w.parse_prompt_weights('A (blue shirt:1.3).')
    cond,_=traced(clean)
    assert w.phrase_weights('A (blue shirt:1.3).',cond[0][1][w.TRACE])==w.span_weights(cond[0][1][w.TRACE],spans)


def test_inline_and_editor_phrases_scale_all_tokens_and_preserve_vision():
    clean,spans=w.parse_prompt_weights('A (blue shirt:1.3).')
    cond,original=traced(clean,spans)
    out=w.BFSH3TokenWeightEditor().edit(cond,None,mode='embedding magnitude (experimental)',inspect_only=False,weighted_phrases='(blue:0.8)')['result'][0][0][0]
    assert torch.equal(out[:,:3],original[:,:3])
    for i,c in enumerate(clean):
        expected=1.3*.8 if 2<=i<6 else 1.3 if 6<=i<12 else 1.
        torch.testing.assert_close(out[:,3+i],original[:,3+i]*expected)
    # A manually selected token can be explicitly restored to neutral over a phrase default.
    index=clean.index('b')
    out=w.BFSH3TokenWeightEditor().edit(cond,None,mode='embedding magnitude (experimental)',inspect_only=False,weights_json=json.dumps({'weights':{str(index):1.}}))['result'][0][0][0]
    assert torch.equal(out[:,3+index],original[:,3+index])
    assert torch.equal(cond[0][0],original)


def test_inline_weights_are_cleaned_before_native_encoding(monkeypatch):
    seen={}
    class Native:
        @classmethod
        def execute(cls,**kwargs):
            seen.update(kwargs);cond,_=traced(kwargs['prompt'])
            # Simulate the native encoder, without BFS trace metadata.
            cond[0][1].pop(w.TRACE)
            return (cond,{'samples':'latent'})
    monkeypatch.setitem(sys.modules,'comfy_extras.nodes_minimax_h3',SimpleNamespace(MiniMaxH3ReferenceToVideo=Native))
    result=workflow.BFSH3NativeReferenceSetToVideo().encode(clip_with_offsets(),None,torch.zeros(5,32,32,3),(torch.zeros(1,32,32,3),),'motion_transfer: (blue shirt:1.3)',32,32,5,validate_h3_sections=False)
    assert seen['prompt']=='motion_transfer: blue shirt' and result[2]==seen['prompt']
    trace=result[0][0][1][w.TRACE]
    assert w.span_weights(trace,trace['inline_spans'])
    assert seen['ref_images']['ref_image_1'].shape==(1,32,32,3)


def test_no_guessing_when_tokenizer_has_no_exact_offsets():
    cond,_=traced('blue shirt');trace=cond[0][1][w.TRACE]
    for token in trace['tokens']:token.pop('char_start');token.pop('char_end')
    with pytest.raises(ValueError,match='exact tokenizer offsets'):w.phrase_weights('(blue:1.2)',trace)


def test_inspect_stop_has_visible_native_status(monkeypatch):
    class Stop:
        def __init__(self,message):self.message=message
    monkeypatch.setitem(sys.modules,'comfy_execution.graph_utils',SimpleNamespace(ExecutionBlocker=Stop))
    cond,_=traced('blue shirt')
    result=w.BFSH3TokenWeightEditor().edit(cond,None,inspect_only=True)
    assert isinstance(result['result'][0],Stop)
    assert 'inspect_only=false' in result['result'][2]
    assert not w.BFSH3TokenWeightEditor.INPUT_TYPES()['required']['inspect_only'][1]['default']


def test_slow_qwen_offset_companion_is_local_and_checks_exact_ids(monkeypatch):
    class Token:
        special=True
    token=Token();calls=[]
    class Fast:
        def add_special_tokens(self,items):calls.append(items)
        def encode(self,text,**kwargs):return SimpleNamespace(ids=[1,2],offsets=[(0,4),(4,10)])
    def no_offsets(self,*args,**kwargs):raise NotImplementedError('slow tokenizer')
    Slow=type('Qwen2Tokenizer',(),{'__call__':no_offsets,'added_tokens_decoder':{9:token}})
    slow=Slow()
    monkeypatch.setitem(sys.modules,'transformers.convert_slow_tokenizer',SimpleNamespace(convert_slow_tokenizer=lambda backend:Fast()))
    assert w.exact_offsets(slow,'blue shirt',[1,2])==[(0,4),(4,10)]
    assert w.exact_offsets(slow,'blue shirt',[99]) is None
    assert calls==[[token]]
