"""Generic reversible model surgery through native ModelPatcher scaling."""
import fnmatch
import json
import math
import re
from comfy_api.latest import io


def plan_rules(state, rules):
    if not isinstance(rules, list):
        raise ValueError('Rules must be a JSON list')
    planned, counts = [], []
    for rule in rules:
        if not rule.get('enabled', True):
            counts.append(0)
            continue
        pattern = rule.get('pattern', '*')
        regex = re.compile(pattern) if rule.get('regex', False) else None
        scale = float(rule.get('scale', 1.0))
        if not math.isfinite(scale):
            raise ValueError('Scale must be finite')
        count = 0
        for key, value in state.items():
            if not key.endswith(('.weight', '.bias')) or not hasattr(value, 'shape'):
                continue
            if not (regex.search(key) if regex else fnmatch.fnmatchcase(key, pattern)):
                continue
            offset = None
            if 'rows' in rule:
                start, stop = rule['rows']
                if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop <= value.shape[0]:
                    raise ValueError(f'Invalid row range for {key}; use [start, exclusive_stop]')
                offset = (0, start, stop - start)
            count += 1
            if scale != 1:
                planned.append((key, offset, scale))
        counts.append(count)
    return planned, counts


class BFSModelSurgery(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='BFSModelSurgery', display_name='BFS Model Surgery', category='BFS/model',
            description='Inspect and scale arbitrary model weights or row ranges. Rules run in order and overlapping scales multiply. Uses native reversible patches without copying the model. Run once to populate the layer panel. Does not scale CLIP or VAE.',
            inputs=[io.Model.Input('model'), io.String.Input('rules', default='[]', multiline=True,
                tooltip='JSON: [{"pattern":"*final_layer*video_out.weight","scale":0.9}]. Glob or regex:true. Optional rows:[start,exclusive_stop].')],
            outputs=[io.Model.Output(), io.String.Output(display_name='report')])

    @classmethod
    def execute(cls, model, rules='[]'):
        parsed = json.loads(rules)
        state = model.model_state_dict()
        planned, counts = plan_rules(state, parsed)
        result = model.clone()
        for key, offset, scale in planned:
            patch_key = (key, offset) if offset else key
            # strength_patch=0 skips the diff entirely; strength_model scales the
            # materialized weight, including preceding LoRAs. No full-size delta allocation.
            patches = {patch_key: ('diff', (None,))}
            accepted = result.add_patches(patches, strength_patch=0.0, strength_model=scale)
            if patch_key not in accepted:
                raise ValueError(f'Model rejected surgery patch: {key}')
        inventory = [{'key': key, 'shape': list(value.shape)} for key, value in state.items()
                     if key.endswith(('.weight', '.bias')) and hasattr(value, 'shape')]
        report = json.dumps({'parameters': len(inventory), 'rule_matches': counts,
                             'applied_scales': len(planned), 'overlap': 'multiply in rule order'})
        return io.NodeOutput(result, report, ui={'surgery_layers': inventory, 'surgery_report': [report]})


NODE_CLASS_MAPPINGS = {'BFSModelSurgery': BFSModelSurgery}
NODE_DISPLAY_NAME_MAPPINGS = {'BFSModelSurgery': 'BFS Model Surgery'}
