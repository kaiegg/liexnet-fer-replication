import os
import torch
from models.liexnet import LiExNet


def export_to_onnx():
    weights_path = "weights/best_liexnet.pth"
    onnx_output_path = "weights/liexnet.onnx"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found at {weights_path}")

    device = torch.device("cpu")  # Exporting on CPU ensures target framework portability

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    classes = checkpoint.get("classes", [])
    num_classes = len(classes) if classes else 7

    print(
        f"Loading best model checkpoint (Val Acc: {checkpoint.get('val_acc', 0)*100:.2f}%)..."
    )

    # Initialize model architecture and load weights
    model = LiExNet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy input tensor matching LiExNet expected input shape: [batch_size, channels, height, width]
    dummy_input = torch.randn(1, 3, 128, 128, device=device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },  # Enables flexible batch sizes during edge inference
    )

    print(f"★ Model successfully exported to ONNX format: {onnx_output_path}")


if __name__ == "__main__":
    export_to_onnx()
