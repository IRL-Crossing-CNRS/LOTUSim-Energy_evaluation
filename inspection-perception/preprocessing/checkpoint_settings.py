
import torch

#checkpoint_path = "runs/segment/v1_img832/weights/best.pt"
checkpoint_path = "best.pt"
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

print("Completed epoch:", checkpoint.get("epoch"))
print("Training arguments:")
for key, value in checkpoint.get("train_args", {}).items():
    print(f"{key}: {value}")

print("\nStored metrics:")
for key, value in checkpoint.get("train_metrics", {}).items():
    print(f"{key}: {value}")