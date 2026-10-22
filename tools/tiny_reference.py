"""Development-only independent PyTorch CPU oracle for the native tiny graph."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from convert_tiny_model import convert


def verify(checkpoint, folder, cli, env, execute, gpu=False):
    raw, tensors = convert(checkpoint)
    torch.set_num_threads(1)
    values = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    state = {t['name']: values[t['offset']:t['offset'] + t['count']].reshape(t['shape']) for t in tensors}
    def block(x, name):
        return F.prelu(F.conv2d(x, state[name + '.rbr.weight'], state[name + '.rbr.bias'], padding=1), state[name + '.act.weight'])
    def forward(x):
        head = block(x, 'head'); current = head
        for b in range(4):
            feature = block(block(current, f'blocks.{b}.conv1'), f'blocks.{b}.conv2')
            current = current + feature * torch.tanh(feature)
        current = block(current, 'trunk') + head
        current = F.conv2d(current, state['up.0.weight'], state['up.0.bias'], padding=1)
        current = F.prelu(F.pixel_shuffle(current, 2), state['up.2.weight'])
        current = F.conv2d(current, state['up.3.weight'], state['up.3.bias'], padding=1)
        bypass = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        return (bypass * state['skip_scale'] + state['skip_bias'] + current * state['res_gain']).clamp(0, 1)
    results = []
    model = folder / 'tiny-weights.bin'; model.write_bytes(raw)
    for w, h in [(7, 5), (13, 9)]:
        for hardware in ([False, True] if gpu else [False]):
            # Nonuniform, deterministic half inputs exercise padding, residuals,
            # negative activations, pixel-shuffle phase and bicubic edge clamping.
            input = ((torch.arange(w*h*4).reshape(h,w,4)*17 % 101).float()/100).half()
            input[:,:,3] = 1
            expected = forward(input[:,:,:3].float().permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0)
            source, target = folder / 'tiny-input.bin', folder / f'tiny-{w}-{h}-{hardware}.bin'
            source.write_bytes(input.numpy().tobytes())
            config = folder / 'tiny-recipe.json'
            config.write_text(json.dumps({'width':w,'height':h,'output_width':w*2,'output_height':h*2,
                'spatial_only':True,'warp':not hardware,'debug':True,'tiny_model_path':str(model),
                'pipeline':{'ProcessingChain':True,'AdditionalScaler':3,'PipelineMonitoring':0}}))
            receipt_path = target.with_suffix('.json')
            execute([cli,'--config',config,'--input',source,'--output',target,'--receipt',receipt_path],cwd=folder,env=env)
            receipt = json.loads(receipt_path.read_text())
            actual = torch.frombuffer(bytearray(target.read_bytes()),dtype=torch.float16).reshape(h*2,w*2,4).float()
            error = (actual[:,:,:3]-expected).abs()
            if not torch.isfinite(actual).all() or error.max() > .0015 or not (actual[:,:,3] == 1).all():
                raise RuntimeError(f'Tiny numerical mismatch {w}x{h}, hardware={hardware}: max={error.max()}')
            if receipt['runtime']['additional_scaler'] != 'Tiny native 2x (experimental)':
                raise RuntimeError('Tiny graph was bypassed')
            results.append({'width':w,'height':h,'hardware':hardware,'max_abs_error':error.max().item(),
                'mean_abs_error':error.mean().item(),'alpha_exact':True})
    return {'passed':True,'oracle':'PyTorch CPU conv/prelu/pixel_shuffle/bicubic, development only',
        'runtime_python_required':False,'cases':results}
