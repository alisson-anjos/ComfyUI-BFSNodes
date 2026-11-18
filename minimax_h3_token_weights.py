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
MIN_GAIN=-2.0
MAX_GAIN=3.0
_OFFSET_TOKENIZERS=weakref.WeakKeyDictionary()
_BLEND_RE=re.compile(r'\[([^\[\]]*\|[^\[\]]*)\]')
_BLEND_OPT_RE=re.compile(r'(.*?)(?::(-?\d*\.?\d+))?\s*$',re.S)


def expand_prompt_blends(text):
    """Expand [a|b:-1] prompt blending groups into (prompt, logit_bias) variants."""
    matches=list(_BLEND_RE.finditer(text))
    if not matches:return [(text,0.0)]
    groups=[]
    for m in matches:
        opts=[]
        for raw in m.group(1).split('|'):
            opt,bias=_BLEND_OPT_RE.fullmatch(raw.strip()).groups()
            opts.append((opt,float(bias) if bias is not None else 0.0))
        groups.append(opts)
    total=max(len(g) for g in groups)
    variants=[]
    for i in range(total):
        picks=[g[i%len(g)] for g in groups]
        it=iter(picks)
        expanded=_BLEND_RE.sub(lambda _:next(it)[0],text)
        variants.append((expanded,sum(b for _,b in picks)))
    return variants


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
                    if not math.isfinite(gain) or not MIN_GAIN<=gain<=MAX_GAIN:
                        raise ValueError(f'Inline prompt forces must be finite and between {MIN_GAIN:g} and {MAX_GAIN:g}')
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
    if any(not MIN_GAIN<=gain<=MAX_GAIN for gain in result.values()):
        raise ValueError(f'Combined nested/overlapping phrase forces must stay between {MIN_GAIN:g} and {MAX_GAIN:g}')
    return result


def phrase_weights(text,trace):
    if not text.strip():return {}
    clean,spans=parse_prompt_weights(text)
    if not spans:
        if clean==trace['prompt']:return {}
        raise ValueError('Use explicit weighted phrases such as (golden fur:1.3)')
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
        if not 0<=index<len(trace['tokens']) or not math.isfinite(gain) or not MIN_GAIN<=gain<=MAX_GAIN:
            raise ValueError(f'Token indices must exist and forces must be finite between {MIN_GAIN:g} and {MAX_GAIN:g}')
        result[index]=gain
    return result


def attention_biases(layout,plan,control,device):
    text_stop=next(b for a,b,kind in layout.segments if kind=='text')
    if control['sequence_length']!=text_stop:
        raise ValueError('Token attention weights belong to a different conditioning length; connect this editor MODEL and positive together')
    strength=float(control.get('strength',1.0))
    rows=[]
    for start,stop,gain in control['ranges']:
        bias=strength*math.log(gain if gain>0.0 else 0.05)
        rows.append((start,stop,torch.full((plan.target_stop-plan.target_start,),bias,device=device)))
    for start,stop,bias in control.get('extra_biases',()):
        rows.append((start,stop,torch.full((plan.target_stop-plan.target_start,),float(bias),device=device)))
    return rows


def _emit_token_impact(q,k,start,num_tokens,seq_len,prompt_sha256):
    if num_tokens<=0 or k.shape[-2]<start+num_tokens or q.shape[-2]<=seq_len:
        return False
    try:
        from server import PromptServer
        srv=getattr(PromptServer,'instance',None)
        if srv is None:return False
        with torch.no_grad():
            q_sub=q[:,:,seq_len::32,:].float()
            k_sub=k[:,:,start:start+num_tokens,:].float()
            if q_sub.shape[2]==0:return False
            scores=torch.matmul(q_sub,k_sub.transpose(-2,-1))/math.sqrt(q_sub.shape[-1])
            probs=scores.softmax(dim=-1).mean(dim=(0,1,2))
            peak=float(probs.max().item())
            if peak<=0.0:return False
            impacts=[round(float(x)/peak,3) for x in probs.detach().cpu().tolist()]
        srv.send_sync('bfs_h3_token_impact',{'prompt_sha256':prompt_sha256,'impacts':impacts})
        return True
    except Exception:
        return False


def _direct_attention_override(ranges,strength,previous,extra_biases=(),probe_state=None):
    def override(func,q,k,v,heads,mask=None,**kwargs):
        if probe_state is not None and not probe_state['done']:
            probe_state['calls']+=1
            if probe_state['calls']>=probe_state['target_call']:
                if _emit_token_impact(q,k,probe_state['start'],probe_state['num_tokens'],probe_state['seq_len'],probe_state['prompt_sha256']):
                    probe_state['done']=True
        k_len=k.shape[-2]
        valid=[(a,b,g) for a,b,g in ranges if b<=k_len]
        valid_extra=[(a,b,bias) for a,b,bias in extra_biases if b<=k_len and bias!=0.0]
        if valid or valid_extra:
            v_scaled=None
            bias_mask=None
            def ensure_mask():
                nonlocal bias_mask
                if bias_mask is None:
                    if mask is None:bias_mask=q.new_zeros((1,1,1,k_len))
                    elif mask.dtype!=torch.bool:bias_mask=mask.clone()
                return bias_mask
            for start,stop,gain in valid:
                if gain<0.0:
                    if v_scaled is None:v_scaled=v.clone()
                    v_scaled[...,start:stop,:]*=(gain*strength)
                elif gain<1.0:
                    if v_scaled is None:v_scaled=v.clone()
                    v_scaled[...,start:stop,:]*=gain
                if gain>0.0:
                    bias=strength*math.log(gain)
                    if bias!=0.0 and ensure_mask() is not None:
                        bias_mask[...,start:stop]+=bias
            for start,stop,bias in valid_extra:
                if ensure_mask() is not None:
                    bias_mask[...,start:stop]+=bias
            if v_scaled is not None:v=v_scaled
            if bias_mask is not None:mask=bias_mask
        if previous is not None:
            return previous(func,q,k,v,heads,mask=mask,**kwargs)
        return func(q,k,v,heads,mask=mask,**kwargs)
    return override


def _in_step_window(model,args,start_percent,end_percent):
    if start_percent<=0.0 and end_percent>=1.0:
        return True
    try:
        ms=model.get_model_object('model_sampling')
    except Exception:
        ms=None
    if ms is None:
        return True
    sigma=float(args['timestep'].flatten()[0])
    if start_percent>0.0 and sigma>float(ms.percent_to_sigma(start_percent))+1e-5:
        return False
    if end_percent<1.0 and sigma<float(ms.percent_to_sigma(end_percent))-1e-5:
        return False
    return True


def _install_direct_attention(model,ranges,strength,start_percent=0.0,end_percent=1.0,extra_biases=(),probe_cfg=None):
    previous=model.model_options.get('model_function_wrapper')
    probe_state=dict(probe_cfg,calls=0,done=False,target_call=12) if probe_cfg else None
    def wrapper(model_function,args):
        conds=args.get('c',{})
        to=dict(conds.get('transformer_options',{}))
        cond_or_uncond=to.get('cond_or_uncond')
        if cond_or_uncond is None or 0 in cond_or_uncond:
            active=(bool(ranges or extra_biases) and _in_step_window(model,args,start_percent,end_percent))
            need_probe=(probe_state is not None and not probe_state['done'])
            if active or need_probe:
                eff_ranges=ranges if active else ()
                eff_extra=extra_biases if active else ()
                to['optimized_attention_override']=_direct_attention_override(
                    eff_ranges,strength,to.get('optimized_attention_override'),
                    extra_biases=eff_extra,probe_state=probe_state if need_probe else None,
                )
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


def _grouped_token_phrases(trace,weights):
    tokens=trace['tokens'];prompt=trace.get('prompt','')
    groups=[]
    for index,gain in sorted(weights.items()):
        if groups and groups[-1][1]+1==index and groups[-1][2]==gain:
            groups[-1]=(groups[-1][0],index,gain)
        else:
            groups.append((index,index,gain))
    items=[]
    for a,b,gain in groups:
        if all('char_start' in tokens[i] and 'char_end' in tokens[i] for i in range(a,b+1)):
            ca,cb=tokens[a]['char_start'],tokens[b]['char_end']
            raw=prompt[ca:cb]
        else:
            ca=cb=None
            raw=''.join(tokens[i]['label'] for i in range(a,b+1))
        lead=raw[:len(raw)-len(raw.lstrip())]
        trail=raw[len(raw.rstrip()):]
        core=raw.strip() or raw
        items.append((a,b,ca,cb,lead,core,trail,gain))
    return items


def format_report(trace,weights,mode,attention_strength,base_weights,report_format='json'):
    if report_format=='compact (changes)':
        tag=f'attn x{float(attention_strength):g}' if mode=='attention logits (experimental)' else 'embed'
        if not weights:return f'{tag} | baseline (no overrides)'
        parts=[f'({core}:{gain:g})' for _,_,_,_,_,core,_,gain in _grouped_token_phrases(trace,weights)]
        return f'{tag} | ' + ' '.join(parts)
    if report_format=='annotated prompt':
        prompt=trace.get('prompt','')
        items=_grouped_token_phrases(trace,weights)
        if items and all(ca is not None for _,_,ca,cb,_,_,_,_ in items):
            out=[];cursor=0
            for _,_,ca,cb,lead,core,trail,gain in items:
                out.append(prompt[cursor:ca])
                out.append(f'{lead}({core}:{gain:g}){trail}')
                cursor=cb
            out.append(prompt[cursor:])
            return ''.join(out)
        # Fallback when character offsets are unavailable
        by_idx={a:(b,lead,core,trail,gain) for a,b,_,_,lead,core,trail,gain in items}
        out=[];i=0;tokens=trace['tokens']
        while i<len(tokens):
            if i in by_idx:
                b,lead,core,trail,gain=by_idx[i]
                out.append(f'{lead}({core}:{gain:g}){trail}')
                i=b+1
            else:
                out.append(tokens[i]['label']);i+=1
        return ''.join(out)
    return json.dumps(dict(mode=mode,attention_strength=float(attention_strength),weighted_tokens=len(weights),forces=weights,phrase_weighted_tokens=len(base_weights)),indent=2)


def _apply_prompt_blends(conditioning,clip,blended_text,trace):
    variants=expand_prompt_blends(blended_text)
    if len(variants)<=1:
        return conditioning,(),{}
    if clip is None:
        raise ValueError('Prompt blending [a|b] requires H3 CLIP to be connected')
    tensor,meta=conditioning[0]
    tags=meta.get('minimax_token_tags')
    if tags is None:
        raise ValueError('Native H3 token tags are required for prompt blending')
    decoder=clip.tokenizer.qwen3vl_32b
    new_tensors=[];new_tags=[];new_records=[];extra_biases=[];blend_weights={};prompt_parts=[];char_offset=0
    for idx,(opt_text,opt_bias) in enumerate(variants):
        clean_opt,opt_spans=parse_prompt_weights(opt_text)
        tokens=clip.tokenize(clean_opt)
        batches=tokens.get('qwen3vl_32b')
        if not batches or len(batches)!=1:
            raise ValueError('Prompt blending requires the native H3 Qwen3-VL tokenizer')
        ids=[int(t[0]) for t in batches[0]]
        n=len(ids)
        if idx==0 and clean_opt==trace['prompt']:
            opt_tensor,opt_tags=tensor,tags.reshape(-1)
            seg_start=trace['start']
        else:
            enc=clip.encode_from_tokens_scheduled(tokens)[0]
            enc_t,enc_m=enc[0],enc[1]
            enc_tags=enc_m['minimax_token_tags'].reshape(-1)
            if idx==0:
                opt_tensor,opt_tags=enc_t,enc_tags
                seg_start=enc_t.shape[1]-n
            else:
                opt_tensor,opt_tags=enc_t[:,enc_t.shape[1]-n:],enc_tags[enc_tags.shape[0]-n:]
                seg_start=sum(t.shape[1] for t in new_tensors)
        new_tensors.append(opt_tensor);new_tags.append(opt_tags)
        offsets=exact_offsets(getattr(decoder,'tokenizer',None),clean_opt,ids)
        words=list(re.finditer(r'\S+',clean_opt))
        base_tok_idx=len(new_records)
        for ti,tid in enumerate(ids):
            label=decoder.decode([tid],skip_special_tokens=False)
            word=label
            if offsets:
                a,b=offsets[ti]
                match=next((m for m in words if m.start()<b and m.end()>a),None)
                if match:word=match.group()
            rec=dict(index=base_tok_idx+ti,token_id=tid,label=label,word=word)
            if offsets:rec.update(char_start=char_offset+int(offsets[ti][0]),char_end=char_offset+int(offsets[ti][1]))
            new_records.append(rec)
        for sp in opt_spans:
            for ti in range(n):
                rec=new_records[base_tok_idx+ti]
                if 'char_start' in rec and rec['char_start']-char_offset<sp['end'] and rec['char_end']-char_offset>sp['start']:
                    k=rec['index'];blend_weights[k]=blend_weights.get(k,1.)*sp['gain']
        if opt_bias!=0.0:
            extra_biases.append((seg_start,seg_start+n,float(opt_bias)))
        prompt_parts.append(clean_opt);char_offset+=len(clean_opt)+3
    joined_prompt=' | '.join(prompt_parts)
    merged_tensor=torch.cat(new_tensors,dim=1)
    merged_tags=torch.cat(new_tags,dim=0)
    new_trace=dict(prompt_sha256=hashlib.sha256(joined_prompt.encode()).hexdigest(),prompt=joined_prompt,
                   sequence_length=merged_tensor.shape[1],start=trace['start'],tokens=new_records,inline_spans=[])
    new_meta=dict(meta,minimax_token_tags=merged_tags,**{TRACE:new_trace})
    return [[merged_tensor,new_meta]],tuple(extra_biases),blend_weights


class BFSH3TokenWeightEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'conditioning':('CONDITIONING',),'model':('MODEL',),
            'mode':(['attention logits (experimental)','embedding magnitude (experimental)'],{'tooltip':'attention logits (recommended): modifies attention logits (log(gain) * attention_strength) and scales V for gain < 1.0, bypassing DiT RMSNorm for strong control. embedding magnitude: scales raw input embeddings before DiT (often absorbed by RMSNorm).'}),
            'inspect_only':('BOOLEAN',{'default':False,'tooltip':'On: populate the token panel and stop before sampling (no GPU render). Switch off and queue again to generate.'}),
            'weights_json':('STRING',{'default':'{}','multiline':True,'tooltip':'JSON weight map managed by the token panel. Automatically resets if the upstream prompt changes.'})},
            'optional':{'clip':('CLIP',),'original_prompt':('STRING',{'forceInput':True,'tooltip':'Exact clean prompt sent to the upstream text encoder. Used with CLIP to map token positions without re-encoding.'}),
                'weighted_phrases':('STRING',{'default':'','multiline':True,'tooltip':'Synced with the panel. Supports (phrase:weight) from -2.0 to 3.0 (>1 boost, 0.05..0.99 suppress logits+V, 0 zero V, <0 invert V) and [a|b:bias] blending. Does not re-run Qwen3-VL unless [a|b] blending is used.'}),
                'attention_strength':('FLOAT',{'default':1.0,'min':0.0,'max':5.0,'step':0.1,'tooltip':'Global sensitivity multiplier on attention logit biases (attention_strength * log(gain)). Does not affect tokens at 1.0.'}),
                'report_format':(['json','compact (changes)','annotated prompt'],{'default':'json','tooltip':'report output format: json (raw indices), compact (changes only, ideal for video overlay labels), or annotated prompt (full text with (word:weight)).'}),
                'start_percent':('FLOAT',{'default':0.0,'min':0.0,'max':1.0,'step':0.01,'tooltip':'Start of the sampling range where token attention weights apply (0.0 = first step). Keep 0.0..1.0 for full generation.'}),
                'end_percent':('FLOAT',{'default':1.0,'min':0.0,'max':1.0,'step':0.01,'tooltip':'End of the sampling range where token attention weights apply (1.0 = last step).'})}}
    RETURN_TYPES=('CONDITIONING','MODEL','STRING')
    RETURN_NAMES=('positive','model','report')
    FUNCTION='edit';CATEGORY='BFS/MiniMax H3'
    DESCRIPTION=('Post-encoder per-token/phrase weight editor for MiniMax H3 (T2V, I2V/FL2VA, Ref2V, Image, and TARA). '
        'Works without re-running Qwen3-VL: upstream encodes the clean prompt once, and this node maps token positions via CPU tokenizer '
        'and applies weights in DiT attention (or Flash Partitions with TARA). '
        'Weights (-2.0..3.0): >1.0 boosts attention logits; 0.05..0.99 lowers logits and scales V; 0.0 zeros V (neutralizes); <0.0 inverts V (pushes away). '
        'Panel tips: click a token, Ctrl/Cmd-click to multi-select, double-click to select all occurrences of a word. Visual tokens are never modified.')

    def edit(self,conditioning,model,mode='attention logits (experimental)',inspect_only=False,weights_json='{}',clip=None,original_prompt=None,weighted_phrases='',attention_strength=1.0,report_format='json',start_percent=0.0,end_percent=1.0):
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
        extra_biases=()
        prompt_changed=False
        try:
            raw_w=json.loads(weights_json)
            if isinstance(raw_w,dict) and raw_w.get('prompt_sha256') and raw_w['prompt_sha256']!=trace['prompt_sha256']:
                prompt_changed=True
        except (ValueError,TypeError):
            pass
        has_blend=(not prompt_changed) and bool(_BLEND_RE.search(weighted_phrases))
        try:
            if has_blend:
                conditioning,extra_biases,base_weights=_apply_prompt_blends(conditioning,clip,weighted_phrases,trace)
                tensor,meta=conditioning[0];trace=meta[TRACE]
            else:
                base_weights=span_weights(trace,trace.get('inline_spans',[]))
                if not prompt_changed:
                    extra=phrase_weights(weighted_phrases,trace)
                    for index,gain in extra.items():base_weights[index]=base_weights.get(index,1.)*gain
            if any(not MIN_GAIN<=g<=MAX_GAIN for g in base_weights.values()):raise ValueError(f'Combined phrase forces must stay between {MIN_GAIN:g} and {MAX_GAIN:g}')
            weights=dict(base_weights)
            if not prompt_changed:
                weights.update(parse_weights(weights_json,trace))
            weights={i:g for i,g in weights.items() if g!=1}
        except ValueError as exc:weights={};error=str(exc)
        rows=[trace['start']+i for i in range(len(trace['tokens']))]
        tags=meta.get('minimax_token_tags')
        if tags is None or not bool(tags.reshape(-1)[rows].eq(1).all()):
            raise ValueError('Token trace overlaps visual or invalid rows; refusing to weight those embeddings')
        norms=tensor[0,rows].detach().float().norm(dim=-1).cpu().tolist()
        records=[dict(r,norm=norms[i],gain=weights.get(i,1.),base_gain=base_weights.get(i,1.)) for i,r in enumerate(trace['tokens'])]
        ui={'bfs_h3_tokens':[dict(tokens=records,prompt_sha256=trace['prompt_sha256'],prompt=trace['prompt'],prompt_changed=prompt_changed,mode=mode,
              message=error or ('Inspect only: sampling blocked. Adjust weights, switch inspect_only off, queue again.' if inspect_only else ('Prompt changed: stale overrides were reset for this run.' if prompt_changed else 'Weights applied for this queued execution.')))]}
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
            spans=contiguous_ranges(weights,trace['start']) if weights else []
            if model is not None and (weights or extra_biases or not model.model_options.get('bfs_h3_native_temporal_v1')):
                model=model.clone();opts=dict(model.model_options.get('transformer_options',{}))
                if OPTION in opts:
                    raise ValueError('Use only one Token Weight Editor per MODEL path')
                if weights or extra_biases:
                    opts[OPTION]=dict(sequence_length=trace['sequence_length'],ranges=spans,extra_biases=extra_biases,strength=float(attention_strength),start_percent=float(start_percent),end_percent=float(end_percent))
                    model.model_options['transformer_options']=opts
                if not model.model_options.get('bfs_h3_native_temporal_v1'):
                    probe_cfg=dict(start=trace['start'],num_tokens=len(trace['tokens']),seq_len=trace['sequence_length'],prompt_sha256=trace['prompt_sha256'])
                    _install_direct_attention(model,spans,float(attention_strength),start_percent=float(start_percent),end_percent=float(end_percent),extra_biases=extra_biases,probe_cfg=probe_cfg)
        else:raise ValueError('Unknown token weight mode')
        print(f'[BFS H3 Token Editor] sampling enabled; mode={mode}; {len(weights)} weighted text token(s).',flush=True)
        report=format_report(trace,weights,mode,attention_strength,base_weights,report_format=report_format)
        return {'ui':ui,'result':(conditioning,model,report)}


NODE_CLASS_MAPPINGS={'BFSH3TokenWeightEditor':BFSH3TokenWeightEditor}
NODE_DISPLAY_NAME_MAPPINGS={'BFSH3TokenWeightEditor':'BFS H3 Token / Word Weight Editor (Experimental)'}
