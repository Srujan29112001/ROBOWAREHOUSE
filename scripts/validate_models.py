"""
Validate model architectures and configurations.

Checks that all models load correctly and have expected shapes.
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robo_vla.perception.dino import DinoV2Detector
from robo_vla.perception.depth import MiDaSDepthEstimator
from robo_vla.perception.gaussian_splatting import GaussianSplatting
from robo_vla.language.llama import LlamaQLoRA
from robo_vla.language.grounding import LanguageGroundingModule
from robo_vla.vla_model.vla import VLAModel
from robo_vla.vla_model.cross_attention import CrossAttentionFusion
from robo_vla.vla_model.action_decoder import ActionDecoder


def validate_perception_models():
    """Validate perception models."""
    print("=" * 60)
    print("VALIDATING PERCEPTION MODELS")
    print("=" * 60)

    # Test DINO v2
    print("\n1. DINO v2 Detector...")
    try:
        dino = DinoV2Detector(device='cpu')
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = dino.detect(test_img)
        assert 'boxes' in result
        assert 'features' in result
        print("   ✅ DINO v2 validation passed")
    except Exception as e:
        print(f"   ❌ DINO v2 validation failed: {e}")
        return False

    # Test MiDaS
    print("\n2. MiDaS Depth Estimator...")
    try:
        midas = MiDaSDepthEstimator(device='cpu')
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = midas.estimate(test_img)
        assert depth.shape == (480, 640)
        print("   ✅ MiDaS validation passed")
    except Exception as e:
        print(f"   ❌ MiDaS validation failed: {e}")
        return False

    # Test Gaussian Splatting
    print("\n3. Gaussian Splatting...")
    try:
        splatter = GaussianSplatting(num_gaussians=100, device='cpu')
        points = np.random.randn(100, 3)
        colors = np.random.rand(100, 3)
        splatter.initialize_from_pointcloud(points, colors)
        assert splatter.means is not None
        print("   ✅ Gaussian Splatting validation passed")
    except Exception as e:
        print(f"   ❌ Gaussian Splatting validation failed: {e}")
        return False

    return True


def validate_language_models():
    """Validate language models."""
    print("\n" + "=" * 60)
    print("VALIDATING LANGUAGE MODELS")
    print("=" * 60)

    # Test Llama
    print("\n1. Llama 3.1 QLoRA...")
    try:
        llama = LlamaQLoRA(device='cpu')
        result = llama.process_command("Pick the red box")
        assert 'intent' in result or 'embeddings' in result
        print("   ✅ Llama validation passed")
    except Exception as e:
        print(f"   ❌ Llama validation failed: {e}")
        return False

    # Test Language Grounding
    print("\n2. Language Grounding Module...")
    try:
        grounding = LanguageGroundingModule(device='cpu')
        result = grounding.process("Move to position")
        assert 'embeddings' in result
        print("   ✅ Language Grounding validation passed")
    except Exception as e:
        print(f"   ❌ Language Grounding validation failed: {e}")
        return False

    return True


def validate_vla_model():
    """Validate VLA model."""
    print("\n" + "=" * 60)
    print("VALIDATING VLA MODEL")
    print("=" * 60)

    # Test Cross-Attention
    print("\n1. Cross-Attention Fusion...")
    try:
        fusion = CrossAttentionFusion()
        visual = torch.randn(1, 10, 768)
        text = torch.randn(1, 20, 4096)
        state = torch.randn(1, 8)
        output, attn = fusion(visual, text, state)
        assert output is not None
        print("   ✅ Cross-Attention validation passed")
    except Exception as e:
        print(f"   ❌ Cross-Attention validation failed: {e}")
        return False

    # Test Action Decoder
    print("\n2. Action Decoder...")
    try:
        decoder = ActionDecoder()
        features = torch.randn(1, 10, 256)
        actions, success = decoder(features, steps=5)
        assert len(actions) == 5
        print("   ✅ Action Decoder validation passed")
    except Exception as e:
        print(f"   ❌ Action Decoder validation failed: {e}")
        return False

    # Test full VLA model
    print("\n3. Complete VLA Model...")
    try:
        vla = VLAModel(device='cpu')
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        command = "Pick the box"
        state = np.zeros(8)
        result = vla.predict(img, command, state)
        assert 'joint_commands' in result
        assert len(result['joint_commands']) == 7
        print("   ✅ VLA Model validation passed")
    except Exception as e:
        print(f"   ❌ VLA Model validation failed: {e}")
        return False

    return True


def validate_model_shapes():
    """Validate output shapes."""
    print("\n" + "=" * 60)
    print("VALIDATING OUTPUT SHAPES")
    print("=" * 60)

    vla = VLAModel(device='cpu')
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = vla.predict(img, "test command", np.zeros(8))

    checks = [
        (len(result['joint_commands']), 7, "Joint commands"),
        (0 <= result['gripper_command'] <= 1, True, "Gripper range"),
        (0 <= result['success_probability'] <= 1, True, "Success prob range"),
    ]

    all_passed = True
    for actual, expected, name in checks:
        if actual == expected:
            print(f"   ✅ {name}: {actual}")
        else:
            print(f"   ❌ {name}: Expected {expected}, got {actual}")
            all_passed = False

    return all_passed


def main():
    """Run all validations."""
    print("\n" + "=" * 60)
    print("MODEL VALIDATION SUITE")
    print("=" * 60)

    results = {
        'Perception Models': validate_perception_models(),
        'Language Models': validate_language_models(),
        'VLA Model': validate_vla_model(),
        'Output Shapes': validate_model_shapes(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:<30} {status}")

    print("=" * 60)

    if all_passed:
        print("\n🎉 All validations passed!")
        return 0
    else:
        print("\n⚠️  Some validations failed. Check logs above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
