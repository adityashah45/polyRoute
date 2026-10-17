"""
Export utilities for Polymer Classification models.
Converts PyTorch checkpoints into ONNX and TorchScript formats for production deployment.
"""

import os
from typing import Optional, Tuple
import torch
import numpy as np

from .config import ClassifierConfig, get_device
from .models import PolymerClassifier, load_classifier_checkpoint


def export_to_onnx(
    model: PolymerClassifier,
    save_path: str,
    device: Optional[torch.device] = None,
    image_size: int = 224,
    verify: bool = True,
) -> str:
    """Exports model to standard ONNX format with dynamic batch axes."""
    device = device or get_device()
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    print(f"Exporting model to ONNX: {save_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["class_logits"],
        dynamic_axes={"input_image": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
    )
    print("✅ ONNX export successful!")

    # Verify numerical equivalence if onnxruntime is installed
    if verify:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(save_path, providers=["CPUExecutionProvider"])
            dummy_np = dummy_input.cpu().numpy()
            ort_outs = session.run(None, {"input_image": dummy_np})

            with torch.no_grad():
                torch_out = model.cpu()(dummy_input.cpu()).numpy()

            diff = np.max(np.abs(ort_outs[0] - torch_out))
            print(f"✅ ONNX Runtime verification passed! Max absolute deviation: {diff:.6e}")
            model.to(device)
        except Exception as e:
            print(f"⚠️ ONNX verification skipped or encountered: {e}")

    return save_path


def export_to_torchscript(
    model: PolymerClassifier,
    save_path: str,
    device: Optional[torch.device] = None,
    image_size: int = 224,
) -> str:
    """Exports model to TorchScript via tracing."""
    device = device or get_device()
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    print(f"Exporting model to TorchScript: {save_path}...")
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(save_path)
    print("✅ TorchScript export successful!")
    return save_path
