"""Inspectable H3 prompt-token provenance and opt-in weighting.

Words are recovered from the actual tokenizer and original prompt, never guessed
by inverting contextual embeddings. Visual rows are excluded by native H3 tags.
"""
import hashlib
import json
import math
import re
import weakref

import torch

TRACE='bfs_h3_prompt_trace'
OPTION='bfs_h3_token_attention'
_OFFSET_TOKENIZERS=weakref.WeakKeyDictionary()


def parse_prompt_weights(text):
    """Explicit (phrase:number) only; ordinary parentheses remain literal.

    Escaped parentheses/colon/backslash are literal. Nested explicit groups
    multiply their token gains later, when character spans map to BPE tokens.
    """
    def parse(source):
        output=[];spans=[];i=0;length=0
        while i<len(source):
            ch=source[i]
            if ch=='\\' and i+1<len(source) and source[i+1] in '():\\':
                output.append(source[i+1]);length+=1;i+=2;continue
            if ch!='(':
                output.append(ch);length+=1;i+=1;continue
            depth=1;j=i+1;colon=None
            while j<len(source) and depth:
                if source[j]=='\\' and j+1<len(source):j+=2;continue
                if source[j]=='(':depth+=1
                elif source[j]==')':depth-=1
                elif source[j]==':' and depth==1:colon=j
                j+=1
            if depth:
                output.append(ch);length+=1;i+=1;continue
            stop=j-1;gain=None
            inside=source[i+1:stop]
            # H3 captions commonly contain (00:03.333) or (Shot 1 at 00:03).
            # Keep timestamp/ratio parentheses literal rather than reading time as a force.
            clock=bool(re.search(r'(?:^|[^\w])\d+:\d{2}(?::\d{2})?(?:\.\d+)?\s*$',inside))
            numeric_body=colon is not None and bool(re.fullmatch(r'[\d\s:+.\-]+',source[i+1:colon]))
            if colon is not None and not clock and not numeric_body:
                suffix=source[colon+1:stop].strip()
                if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',suffix):
                    gain=float(suffix)
                    if not math.isfinite(gain) or not .05<=gain<=3:
                        raise ValueError('Inline prompt forces must be finite and between 0.05 and 3')
            inner,nested=parse(source[i+1:colon] if gain is not None else source[i+1:stop])
            prefix='' if gain is not None else '(';piece=prefix+inner+('' if gain is not None else ')')
            for span in nested:spans.append(dict(start=length+len(prefix)+span['start'],end=length+len(prefix)+span['end'],gain=span['gain']))
            if gain is not None:
                if not inner.strip():raise ValueError('A weighted phrase cannot be empty')
                spans.append(dict(start=length,end=length+len(inner),gain=gain))
            output.append(piece);length+=len(piece);i=j
        return ''.join(output),spans
    return parse(text)


def span_weights(trace,spans):
    if spans and any('char_start' not in t for t in trace['tokens']):
        raise ValueError('Phrase weighting requires exact tokenizer offsets. Use the native H3 fast tokenizer or the visual token editor.')
    result={}
    for span in spans:
        for token in trace['tokens']:
            if token['char_start']<span['end'] and token['char_end']>span['start']:
                index=token['index'];result[index]=result.get(index,1.)*span['gain']
    if any(not .05<=gain<=3 for gain in result.values()):
        raise ValueError('Combined nested/overlapping phrase forces must stay between 0.05 and 3')
    return result


def phrase_weights(text,trace):
    if not text.strip():return {}
    clean,spans=parse_prompt_weights(text)
    if not spans:raise ValueError('Use explicit weighted phrases such as (golden fur:1.3)')
    if clean==trace['prompt']:return span_weights(trace,spans)
    covered=[False]*len(clean)
    for span in spans:
        for i in range(span['start'],span['end']):covered[i]=True
    if ''.join(c for i,c in enumerate(clean) if not covered[i]).strip(' \r\n\t,;'):
        raise ValueError('Weighted phrases must be (exact phrase:force) directives, or a full annotated prompt matching the encoded text')
    mapped=[]
    for span in spans:
        phrase=clean[span['start']:span['end']]
        pattern=(r'(?<!\w)' if phrase[0].isalnum() or phrase[0]=='_' else '')+re.escape(phrase)+(r'(?!\w)' if phrase[-1].isalnum() or phrase[-1]=='_' else '')
        matches=list(re.finditer(pattern,trace['prompt']))
        if not matches:raise ValueError(f'Weighted phrase not present in the encoded prompt: {phrase!r}')
        mapped.extend(dict(start=m.start(),end=m.end(),gain=span['gain']) for m in matches)
    return span_weights(trace,mapped)


def exact_offsets(backend,prompt,ids):
    if backend is None:return None
    try:
        encoded=backend(prompt,add_special_tokens=False,return_offsets_mapping=True)
        if list(encoded['input_ids'])==ids:return encoded['offset_mapping']
    except (TypeError,ValueError,KeyError,NotImplementedError):
        pass
    for candidate in (getattr(backend,'backend_tokenizer',None),getattr(backend,'_tokenizer',None)):
        if candidate is not None:
            encoded=candidate.encode(prompt,add_special_tokens=False)
            if list(encoded.ids)==ids:return encoded.offsets
    # Older ComfyUI uses the slow Qwen2 tokenizer. Build an offset-capable
    # companion locally from its exact vocabulary/merges; never download one.
    if type(backend).__name__!='Qwen2Tokenizer':return None
    try:
        fast=_OFFSET_TOKENIZERS.get(backend)
        if fast is None:
            from transformers.convert_slow_tokenizer import convert_slow_tokenizer
            fast=convert_slow_tokenizer(backend)
            for _,token in sorted(backend.added_tokens_decoder.items()):
                if getattr(token,'special',False):fast.add_special_tokens([token])
                else:fast.add_tokens([token])
            _OFFSET_TOKENIZERS[backend]=fast
        encoded=fast.encode(prompt,add_special_tokens=False)
        if list(encoded.ids)==ids:return encoded.offsets
    except (AttributeError,KeyError,TypeError,ValueError,NotImplementedError):
        pass
    return None


def attach_trace(conditioning,clip,prompt,inline_spans=None):
    tokens=clip.tokenize(prompt)
    batches=tokens.get('qwen3vl_32b')
    if not batches or len(batches)!=1:
        raise ValueError('Prompt tracing requires the native H3 Qwen3-VL tokenizer')
    ids=[int(t[0]) for t in batches[0]]
    decoder=clip.tokenizer.qwen3vl_32b
    offsets=exact_offsets(getattr(decoder,'tokenizer',None),prompt,ids)
    words=list(re.finditer(r'\S+',prompt))
    records=[]
    for i,token in enumerate(ids):
        label=decoder.decode([token],skip_special_tokens=False)
        word=label
        if offsets:
            a,b=offsets[i]
            match=next((m for m in words if m.start()<b and m.end()>a),None)
            if match:word=match.group()
        record=dict(index=i,token_id=token,label=label,word=word)
        if offsets:record.update(char_start=int(offsets[i][0]),char_end=int(offsets[i][1]))
        records.append(record)
    result=[]
    for tensor,meta in conditioning:
        tags=meta.get('minimax_token_tags')
        if tags is None or tensor.ndim!=3 or tensor.shape[0]!=1:
            raise ValueError('Native H3 token tags and a single conditioning sequence are required for safe tracing')
        n=tensor.shape[1];tags=tags.reshape(-1)
        if not ids or len(ids)>n or len(tags)!=n or not bool(tags[n-len(ids):].eq(1).all()):
            raise ValueError('Prompt suffix does not match the native H3 text/vision token layout')
        trace=dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),prompt=prompt,
                   sequence_length=n,start=n-len(ids),tokens=records,inline_spans=inline_spans or [])
        if inline_spans:span_weights(trace,inline_spans)
        result.append([tensor,dict(meta,**{TRACE:trace})])
    return result


def parse_weights(text,trace):
    try:data=json.loads(text)
    except (ValueError,TypeError) as exc:raise ValueError('Token weights must be JSON produced by the editor') from exc
    if not isinstance(data,dict):raise ValueError('Token weights must be a JSON object')
    if data.get('prompt_sha256') and data['prompt_sha256']!=trace['prompt_sha256']:
        raise ValueError('Prompt changed: reset token weights in the editor before applying them')
    if not isinstance(data.get('weights',{}),dict):raise ValueError('weights must map token indices to forces')
    result={}
    for key,value in data.get('weights',{}).items():
        index=int(key);gain=float(value)
        if not 0<=index<len(trace['tokens']) or not math.isfinite(gain) or not .05<=gain<=3:
            raise ValueError('Token indices must exist and forces must be finite between 0.05 and 3')
        result[index]=gain
    return result


def attention_biases(layout,plan,control,device):
    text_stop=next(b for a,b,kind in layout.segments if kind=='text')
    if control['sequence_length']!=text_stop:
        raise ValueError('Token attention weights belong to a different conditioning length; connect this editor MODEL and positive together')
    strength=float(control.get('strength',1.0))
    return [(start,stop,torch.full((plan.target_stop-plan.target_start,),strength*math.log(gain),device=device))
            for start,stop,gain in control['ranges']]


def _direct_attention_override(ranges,strength,previous):
    def override(func,q,k,v,heads,mask=None,**kwargs):
        k_len=k.shape[-2]
        valid=[(a,b,g) for a,b,g in ranges if b<=k_len]
        if valid:
            v_scaled=None
            bias_mask=None
            for start,stop,gain in valid:
                if gain<1.0:
                    if v_scaled is None:v_scaled=v.clone()
                    v_scaled[...,start:stop,:]*=gain
                bias=strength*math.log(gain)
                if bias!=0.0:
                    if bias_mask is None:
                        if mask is None:bias_mask=q.new_zeros((1,1,1,k_len))
                        elif mask.dtype!=torch.bool:bias_mask=mask.clone()
                        else:bias_mask=None
                    if bias_mask is not None:
                        bias_mask[...,start:stop]+=bias
            if v_scaled is not None:v=v_scaled
            if bias_mask is not None:mask=bias_mask
        if previous is not None:
            return previous(func,q,k,v,heads,mask=mask,**kwargs)
        return func(q,k,v,heads,mask=mask,**kwargs)
    return override


def _install_direct_attention(model,ranges,strength):
    previous=model.model_options.get('model_function_wrapper')
    def wrapper(model_function,args):
        conds=args.get('c',{})
        to=dict(conds.get('transformer_options',{}))
        cond_or_uncond=to.get('cond_or_uncond')
        if cond_or_uncond is None or 0 in cond_or_uncond:
            to['optimized_attention_override']=_direct_attention_override(ranges,strength,to.get('optimized_attention_override'))
            args=dict(args,c=dict(conds,transformer_options=to))
        return previous(model_function,args) if previous else model_function(args['input'],args['timestep'],**args['c'])
    model.set_model_unet_function_wrapper(wrapper)


def contiguous_ranges(weights,start):
    result=[]
    for index,gain in sorted(weights.items()):
        row=start+index
        if result and result[-1][1]==row and result[-1][2]==gain:
            a,b,g=result[-1];result[-1]=(a,row+1,g)
        else:result.append((row,row+1,gain))
    if len(result)>16:
        raise ValueError('This prototype supports up to 16 disjoint weighted token ranges; select fewer words or a contiguous phrase')
    return result


class BFSH3TokenWeightEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'conditioning':('CONDITIONING',),'model':('MODEL',),
            'mode':(['attention logits (experimental)','embedding magnitude (experimental)'],{}),
            'inspect_only':('BOOLEAN',{'default':False,'tooltip':'On: show tokens and stop before sampling. Adjust tokens in the panel; switch off and queue again to render.'}),
            'weights_json':('STRING',{'default':'{}','multiline':True})},
            'optional':{'clip':('CLIP',),'original_prompt':('STRING',{'forceInput':True}),
                'weighted_phrases':('STRING',{'default':'','multiline':True,'tooltip':'Exact phrases such as (golden fur:1.3), one per line. Applies to every exact occurrence in the encoded prompt. No extra CLIP pass.'}),
                'attention_strength':('FLOAT',{'default':2.0,'min':0.0,'max':5.0,'step':0.1,'tooltip':'Multiplier on attention logit biases in attention mode (higher = stronger emphasis/suppression).'})}}
    RETURN_TYPES=('CONDITIONING','MODEL','STRING')
    RETURN_NAMES=('positive','model','report')
    FUNCTION='edit';CATEGORY='BFS/MiniMax H3'
    DESCRIPTION=('Experimental word/subword controls. Requires traced BFS native H3 conditioning, or the exact original prompt and H3 CLIP. '
        'Does not decode embeddings back into words. Attention mode works with TARA (Flash Partitions) or directly on MiniMax H3 models (T2V/I2V). '
        'No direct edits to visual rows, reference latents or pooled embeddings. Changes take effect on next queue.')

    def edit(self,conditioning,model,mode='attention logits (experimental)',inspect_only=False,weights_json='{}',clip=None,original_prompt=None,weighted_phrases='',attention_strength=2.0):
        if not conditioning or TRACE not in conditioning[0][1]:
            if clip is None or original_prompt is None:
                raise ValueError('Connect traced BFS Native Reference Set to Video conditioning, or provide H3 CLIP and the exact original prompt. Embeddings alone cannot be decoded into original words.')
            clean,spans=parse_prompt_weights(original_prompt)
            if spans:raise ValueError('Inline syntax must be removed BEFORE encoding. Use BFS Native Reference Set to Video with parse_prompt_weights enabled, or use weighted_phrases on already clean conditioning.')
            conditioning=attach_trace(conditioning,clip,original_prompt)
        if len(conditioning)!=1:
            raise ValueError('Token editor currently supports one H3 conditioning sequence, not scheduled conditioning')
        tensor,meta=conditioning[0];trace=meta[TRACE]
        if tensor.shape[1]!=trace['sequence_length']:
            raise ValueError('Conditioning length no longer matches token trace')
        error=None
        base_weights={}
        try:
            base_weights=span_weights(trace,trace.get('inline_spans',[]))
            extra=phrase_weights(weighted_phrases,trace)
            for index,gain in extra.items():base_weights[index]=base_weights.get(index,1.)*gain
            if any(not .05<=g<=3 for g in base_weights.values()):raise ValueError('Combined phrase forces must stay between 0.05 and 3')
            weights=dict(base_weights);weights.update(parse_weights(weights_json,trace))
            weights={i:g for i,g in weights.items() if g!=1}
        except ValueError as exc:weights={};error=str(exc)
        rows=[trace['start']+i for i in range(len(trace['tokens']))]
        tags=meta.get('minimax_token_tags')
        if tags is None or not bool(tags.reshape(-1)[rows].eq(1).all()):
            raise ValueError('Token trace overlaps visual or invalid rows; refusing to weight those embeddings')
        norms=tensor[0,rows].detach().float().norm(dim=-1).cpu().tolist()
        records=[dict(r,norm=norms[i],gain=weights.get(i,1.),base_gain=base_weights.get(i,1.)) for i,r in enumerate(trace['tokens'])]
        ui={'bfs_h3_tokens':[dict(tokens=records,prompt_sha256=trace['prompt_sha256'],mode=mode,
              message=error or ('Inspect only: sampling blocked. Adjust weights, switch inspect_only off, queue again.' if inspect_only else 'Weights applied for this queued execution.'))]}
        if error or inspect_only:
            from comfy_execution.graph_utils import ExecutionBlocker
            stop=ExecutionBlocker(None)
            message=error or 'INSPECT ONLY: token preview is ready; sampling is intentionally blocked. Set inspect_only=false on this node and queue again to render.'
            print('[BFS H3 Token Editor] '+message,flush=True)
            return {'ui':ui,'result':(stop,stop,message)}
        if mode=='embedding magnitude (experimental)':
            if weights:
                edited=tensor.clone()
                for index,gain in weights.items():edited[:,trace['start']+index,:]*=gain
                conditioning=[[edited,dict(meta)]]
        elif mode=='attention logits (experimental)':
            if weights:
                spans=contiguous_ranges(weights,trace['start'])
                model=model.clone();opts=dict(model.model_options.get('transformer_options',{}))
                if OPTION in opts:
                    raise ValueError('Use only one Token Weight Editor per MODEL path')
                opts[OPTION]=dict(sequence_length=trace['sequence_length'],ranges=spans,strength=float(attention_strength))
                model.model_options['transformer_options']=opts
                if not model.model_options.get('bfs_h3_native_temporal_v1'):
                    _install_direct_attention(model,spans,float(attention_strength))
        else:raise ValueError('Unknown token weight mode')
        print(f'[BFS H3 Token Editor] sampling enabled; mode={mode}; {len(weights)} weighted text token(s).',flush=True)
        report=json.dumps(dict(mode=mode,attention_strength=float(attention_strength),weighted_tokens=len(weights),forces=weights,phrase_weighted_tokens=len(base_weights)),indent=2)
        return {'ui':ui,'result':(conditioning,model,report)}


NODE_CLASS_MAPPINGS={'BFSH3TokenWeightEditor':BFSH3TokenWeightEditor}
NODE_DISPLAY_NAME_MAPPINGS={'BFSH3TokenWeightEditor':'BFS H3 Token / Word Weight Editor (Experimental)'}
