import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from scOT.model import ScOT
from scOT.problems.base import get_dataset

def _batch_vrmse(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute VRMSE per sample, then average over the batch.

    VRMSE = sqrt(MSE(pred, target)) / (std(target) + eps),
    where MSE/std are computed over all non-batch dimensions.
    """
    # Flatten non-batch dims
    B = target.shape[0]
    pred_flat = prediction.reshape(B, -1)
    tgt_flat  = target.reshape(B, -1)

    rmse_per = torch.sqrt(torch.mean((pred_flat - tgt_flat) ** 2, dim=1))
    std_per  = torch.std(tgt_flat, dim=1, unbiased=False)
    vrmse_per = rmse_per / (std_per + eps)
    return vrmse_per.mean()

def evaluate_l1_and_vrmse(args):
    """
    Loads a model and dataset, then calculates the mean L1 loss and VRMSE
    for single-frame predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.model_path}")
    model = ScOT.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    dataset_kwargs = {
        "time_step_size": 1,
        "max_num_time_steps": 1,
    }
    print(f"Loading '{args.dataset_name}' test set from: {args.data_path}")
    dataset = get_dataset(
        dataset=args.dataset_name,
        which="test",
        num_trajectories=-1,
        data_path=args.data_path,
        **dataset_kwargs
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    total_l1_loss = 0.0
    total_vrmse = 0.0
    batch_count = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating L1 & VRMSE"):
            inputs = batch["pixel_values"].to(device, non_blocking=True)
            ground_truth = batch["labels"].to(device, non_blocking=True)
            time = batch["time"].to(device, non_blocking=True)

            prediction = model(pixel_values=inputs, time=time).output

            if hasattr(dataset, 'remove_padding'):
                prediction = dataset.remove_padding(prediction)
                ground_truth = dataset.remove_padding(ground_truth)

            l1_loss = F.l1_loss(prediction, ground_truth)
            vrmse = _batch_vrmse(prediction, ground_truth)

            total_l1_loss += l1_loss.item()
            total_vrmse  += vrmse.item()
            batch_count += 1

    mean_l1 = total_l1_loss / batch_count
    mean_vrmse = total_vrmse / batch_count
    print("\n" + "="*30)
    print("Evaluation Complete")
    print(f"Mean L1 Loss: {mean_l1:.8f}")
    print(f"Mean VRMSE  : {mean_vrmse:.8f}")
    print("="*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A minimal script to calculate the mean L1 and VRMSE for 1-frame forward predictions."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    evaluate_l1_and_vrmse(args)