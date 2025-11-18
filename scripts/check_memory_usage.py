"""
Check memory usage of models.

Analyzes model sizes and VRAM requirements for RTX 3060 (12GB).
"""
import sys
from pathlib import Path
import torch
import numpy as np
import psutil
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robo_vla.perception.dino import DinoV2Detector
from robo_vla.perception.depth import MiDaSDepthEstimator
from robo_vla.language.llama import LlamaQLoRA
from robo_vla.vla_model.vla import VLAModel


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def check_model_memory(name, model_class, device='cpu', **kwargs):
    """Check memory usage for a model."""
    print(f"\n{name}")
    print("-" * 60)

    # Get baseline memory
    if device == 'cuda':
        if not torch.cuda.is_available():
            print("   ⚠️  GPU not available, skipping")
            return None
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
    else:
        process = psutil.Process(os.getpid())
        baseline = process.memory_info().rss

    # Initialize model
    model = model_class(device=device, **kwargs)

    # Get model size
    model_size = get_model_size_mb(model)
    print(f"   Model parameters: {model_size:.2f} MB")

    # Get memory after loading
    if device == 'cuda':
        current = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        memory_used = (peak - baseline) / 1024 / 1024
        print(f"   GPU memory used: {memory_used:.2f} MB")
    else:
        current = process.memory_info().rss
        memory_used = (current - baseline) / 1024 / 1024
        print(f"   RAM used: {memory_used:.2f} MB")

    # Run inference to measure peak usage
    try:
        if 'Detector' in name or 'Estimator' in name:
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            _ = model.detect(test_img) if hasattr(model, 'detect') else model.estimate(test_img)
        elif 'Llama' in name:
            _ = model.process_command("test command")
        elif 'VLA' in name:
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            _ = model.predict(img, "test", np.zeros(8))

        if device == 'cuda':
            peak_after_inference = torch.cuda.max_memory_allocated()
            inference_memory = (peak_after_inference - baseline) / 1024 / 1024
            print(f"   Peak memory (with inference): {inference_memory:.2f} MB")
    except Exception as e:
        print(f"   ⚠️  Inference test failed: {e}")

    return memory_used


def main():
    """Check memory usage for all models."""
    print("=" * 60)
    print("MODEL MEMORY USAGE ANALYSIS")
    print("=" * 60)
    print("\nTarget: RTX 3060 (12GB VRAM)")
    print("=" * 60)

    device = 'cpu'  # Change to 'cuda' if GPU available

    models = [
        ("DINO v2 Detector", DinoV2Detector, {}),
        ("MiDaS Depth Estimator", MiDaSDepthEstimator, {}),
        ("Llama 3.1 8B QLoRA", LlamaQLoRA, {}),
        ("Complete VLA Model", VLAModel, {}),
    ]

    total_memory = 0
    results = {}

    for name, model_class, kwargs in models:
        try:
            memory = check_model_memory(name, model_class, device, **kwargs)
            if memory:
                total_memory += memory
                results[name] = memory
        except Exception as e:
            print(f"\n{name}")
            print("-" * 60)
            print(f"   ❌ Failed to load: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("MEMORY SUMMARY")
    print("=" * 60)

    for name, memory in results.items():
        print(f"{name:<30} {memory:>10.2f} MB")

    print("-" * 60)
    print(f"{'TOTAL':<30} {total_memory:>10.2f} MB")

    # Check if fits in RTX 3060
    rtx_3060_vram = 12 * 1024  # 12GB in MB
    if device == 'cuda':
        if total_memory < rtx_3060_vram:
            margin = rtx_3060_vram - total_memory
            print(f"\n✅ Fits in RTX 3060 with {margin:.2f} MB margin ({margin/rtx_3060_vram*100:.1f}%)")
        else:
            excess = total_memory - rtx_3060_vram
            print(f"\n❌ Exceeds RTX 3060 by {excess:.2f} MB")
            print("   Consider:")
            print("   - Using INT8 quantization")
            print("   - Reducing batch size")
            print("   - Offloading some models to CPU")
    else:
        print(f"\n💡 Run with CUDA to measure GPU memory usage")

    # Generate report
    report_dir = Path(__file__).parent.parent / 'validation-reports'
    report_dir.mkdir(exist_ok=True)

    report_file = report_dir / 'memory_usage.txt'
    with open(report_file, 'w') as f:
        f.write("MODEL MEMORY USAGE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total Memory: {total_memory:.2f} MB\n\n")
        for name, memory in results.items():
            f.write(f"{name}: {memory:.2f} MB\n")

    print(f"\nReport saved to: {report_file}")


if __name__ == '__main__':
    main()
