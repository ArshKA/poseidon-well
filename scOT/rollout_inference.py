import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import json 
import os   

from scOT.model import ScOT
from scOT.problems.base import get_dataset
from scOT.one_frame_inference import _batch_vrmse


def evaluate_l1_and_vrmse(args):

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

    rollout_steps = args.rollout_end_frame - args.rollout_start_frame
    if rollout_steps <= 0:
        raise ValueError("rollout_end_frame must be greater than rollout_start_frame.")
    
    per_frame_metrics = {}

    print(f"Starting evaluation for t=0 rollouts from frame {args.rollout_start_frame} to {args.rollout_end_frame}...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Calculating Rollout L1 & VRMSE")):
            
            is_start_of_sequence = (batch["time"] == 0.0)

            if not is_start_of_sequence.any():
                continue

            initial_frames = batch["pixel_values"][is_start_of_sequence]
            initial_times = batch["time"][is_start_of_sequence]
            
            num_valid_samples_in_batch = initial_frames.shape[0] 
            
            if num_valid_samples_in_batch == 0: # Double-check, should be caught by .any() above
                continue

            current_dataloader_batch_size = batch["pixel_values"].shape[0] # Original batch size from dataloader
            start_index_in_dataset = i * args.batch_size # Base index for the dataloader batch
            
            original_indices_in_batch = torch.arange(start_index_in_dataset, start_index_in_dataset + current_dataloader_batch_size)
            valid_indices_for_gt = original_indices_in_batch[is_start_of_sequence]

            ar_steps_list = [1] * args.rollout_end_frame 
            
            all_ground_truth_rollouts = []
            for sample_idx in valid_indices_for_gt:
                single_rollout = dataset.get_ground_truth_for_rollout(sample_idx.item(), ar_steps_list)
                all_ground_truth_rollouts.append(single_rollout)
            
            ground_truth_rollout = torch.stack(all_ground_truth_rollouts, dim=0).to(device)
            current_frame = initial_frames.to(device, non_blocking=True)
            initial_time_normalized = initial_times

            for step in range(args.rollout_end_frame):
                time_tensor = (initial_time_normalized + step / dataset.constants["time"]).to(device)
                
                prediction = model(pixel_values=current_frame, time=time_tensor).output

                metric_frame_index = step + 1 
                if metric_frame_index >= args.rollout_start_frame:
                    ground_truth = ground_truth_rollout[:, step]
                    
                    if hasattr(dataset, 'remove_padding'):
                        prediction_unpadded = dataset.remove_padding(prediction)
                        ground_truth_unpadded = dataset.remove_padding(ground_truth)
                    else:
                        prediction_unpadded = prediction
                        ground_truth_unpadded = ground_truth
                    
                    l1_loss_batch_mean = F.l1_loss(prediction_unpadded, ground_truth_unpadded).item()
                    vrmse_batch_mean = _batch_vrmse(prediction_unpadded, ground_truth_unpadded).item()

                    if metric_frame_index not in per_frame_metrics:
                        per_frame_metrics[metric_frame_index] = {'l1_sum': 0.0, 'vrmse_sum': 0.0, 'num_batches_contributed': 0}
                    
                    per_frame_metrics[metric_frame_index]['l1_sum'] += l1_loss_batch_mean
                    per_frame_metrics[metric_frame_index]['vrmse_sum'] += vrmse_batch_mean
                    per_frame_metrics[metric_frame_index]['num_batches_contributed'] += 1

                current_frame = prediction
    
    final_metrics_to_save = {}
    
    overall_l1_sum = 0.0
    overall_vrmse_sum = 0.0
    overall_steps_evaluated_count = 0 # Counts how many frame indices actually contributed metrics

    for frame_idx in range(args.rollout_start_frame, args.rollout_end_frame + 1):
        if frame_idx in per_frame_metrics and per_frame_metrics[frame_idx]['num_batches_contributed'] > 0:
            data = per_frame_metrics[frame_idx]
            mean_l1_for_frame = data['l1_sum'] / data['num_batches_contributed']
            mean_vrmse_for_frame = data['vrmse_sum'] / data['num_batches_contributed']
            
            final_metrics_to_save[f"frame_{frame_idx}"] = {
                "mean_l1": mean_l1_for_frame,
                "mean_vrmse": mean_vrmse_for_frame
            }
            
            overall_l1_sum += mean_l1_for_frame
            overall_vrmse_sum += mean_vrmse_for_frame
            overall_steps_evaluated_count += 1 # Count of frames that actually contributed metrics

        else:
            final_metrics_to_save[f"frame_{frame_idx}"] = {
                "mean_l1": 0.0, 
                "mean_vrmse": 0.0
            }


    overall_mean_l1 = overall_l1_sum / overall_steps_evaluated_count if overall_steps_evaluated_count > 0 else 0
    overall_mean_vrmse = overall_vrmse_sum / overall_steps_evaluated_count if overall_steps_evaluated_count > 0 else 0


    print("\n" + "="*30)
    print("Evaluation Complete")
    print(f"Rollout Range: {args.rollout_start_frame} -> {args.rollout_end_frame}")
    print(f"Overall Mean L1 Loss: {overall_mean_l1:.8f}")
    print(f"Overall Mean VRMSE  : {overall_mean_vrmse:.8f}")
    print("="*30)


    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(final_metrics_to_save, f, indent=4)
        print(f"Per-frame metrics saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to calculate rollout errors from the start of each sequence and save per-frame metrics."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model checkpoint.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the root of the dataset.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset class to use.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    
    parser.add_argument("--rollout_start_frame", type=int, required=True, help="The starting frame of the rollout to calculate error (inclusive).")
    parser.add_argument("--rollout_end_frame", type=int, required=True, help="The ending frame of the rollout to calculate error (inclusive).")
    parser.add_argument("--output_path", type=str, default=None, help="Optional: Path to save the per-frame metrics JSON file. E.g., 'results/rollout_metrics.json'")


    args = parser.parse_args()
    evaluate_l1_and_vrmse(args)