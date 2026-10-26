import statistics
import subprocess
import torch
from src.models.vision_encoder import ChestXrayVisionModel
from src.models.fusion import ChestXrayFusionModel

assert torch.cuda.is_available(), 'CUDA is not available'
device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__)

vck = torch.load('checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth', map_location='cpu', weights_only=True)
fck = torch.load('checkpoints/fusion_patient_disjoint_dedup_rho_v1_0/best_model.pth', map_location='cpu', weights_only=True)
vision = ChestXrayVisionModel(num_classes=len(vck['classes']), pretrained=False, use_cbam=vck.get('use_cbam', False), use_se=vck.get('use_se', False))
vision.load_state_dict(vck['model_state_dict'], strict=True)
fusion = ChestXrayFusionModel(num_classes=len(fck['classes']), num_tabular_features=4, pretrained=False, tabular_embedding_dim=64, use_cbam=fck.get('use_cbam', False))
fusion.load_state_dict(fck['model_state_dict'], strict=True)
vision, fusion = vision.to(device).eval(), fusion.to(device).eval()
image = torch.randn(1, 3, 224, 224, device=device)
tabular = torch.randn(1, 4, device=device)

def run_v():
    with torch.inference_mode():
        return vision(image)
def run_f():
    with torch.inference_mode():
        return fusion(image, tabular)

def gpu_snapshot(label):
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw', '--format=csv,noheader'], text=True).strip()
        print(label, out)
    except Exception as e:
        print(label, 'GPU snapshot unavailable:', e)

def measure(fn, iterations=100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations

print('Vision parameters:', sum(p.numel() for p in vision.parameters()))
print('Fusion parameters:', sum(p.numel() for p in fusion.parameters()))
print('Warming up...')
for _ in range(40):
    run_v(); run_f()
torch.cuda.synchronize()
torch.backends.cudnn.benchmark = False
gpu_snapshot('Before timing (temperature, utilization, memory, power):')
vt, ft = [], []
for r in range(15):
    pairs = [('vision', run_v), ('fusion', run_f)]
    if r % 2:
        pairs.reverse()
    for name, fn in pairs:
        ms = measure(fn, 100)
        (vt if name == 'vision' else ft).append(ms)
    print('Round', r + 1, '| vision:', round(vt[-1] if r % 2 == 0 else vt[-1], 3), '| fusion:', round(ft[-1], 3), 'ms/image')
gpu_snapshot('After timing (temperature, utilization, memory, power):')
for name, values in [('Vision', vt), ('Fusion', ft)]:
    print(name, 'median:', round(statistics.median(values), 3), 'ms/image')
    print(name, 'mean:', round(statistics.mean(values), 3), 'ms/image')
    print(name, 'rounds:', [round(x, 3) for x in values])
print('Median latency change (fusion vs vision):', round((statistics.median(ft) / statistics.median(vt) - 1) * 100, 2), '%')
