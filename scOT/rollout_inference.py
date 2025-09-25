import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F

# Make sure these imports point to your project structure
from scOT.model import ScOT
from scOT.problems.base import get_dataset
from scOT.one_frame_inference import _batch_vrmse


def evaluate_l1_and_vrmse(args):
    """
    Loads a model and dataset, then calculates rollout errors starting ONLY from
    the beginning (t=0) of each sequence in a batched and efficient manner.
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

    # Dataloader is set to shuffle=False to make index calculation reliable
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

    total_l1_loss = 0.0
    total_vrmse = 0.0
    steps_evaluated = 0

    print(f"Starting evaluation for t=0 rollouts from frame {args.rollout_start_frame} to {args.rollout_end_frame}...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Calculating Rollout L1 & VRMSE")):
            
            is_start_of_sequence = (batch["time"] == 0.0)

            if not is_start_of_sequence.any():
                continue

            initial_frames = batch["pixel_values"][is_start_of_sequence]
            initial_times = batch["time"][is_start_of_sequence]
            
            current_batch_size = batch["pixel_values"].shape[0]
            start_index = i * args.batch_size
            original_indices = torch.arange(start_index, start_index + current_batch_size)
            valid_indices = original_indices[is_start_of_sequence]

            
            ar_steps_list = [1] * (args.rollout_end_frame)
            all_ground_truth_rollouts = []
            for sample_idx in valid_indices:
                single_rollout = dataset.get_ground_truth_for_rollout(sample_idx.item(), ar_steps_list)
                all_ground_truth_rollouts.append(single_rollout)
            
            ground_truth_rollout = torch.stack(all_ground_truth_rollouts, dim=0).to(device)
            current_frame = initial_frames.to(device, non_blocking=True)
            initial_time_normalized = initial_times

            for step in range(args.rollout_end_frame):
                time_tensor = (initial_time_normalized + step / dataset.constants["time"]).to(device)
                
                prediction = model(pixel_values=current_frame, time=time_tensor).output

                if (step + 1) >= args.rollout_start_frame:
                    ground_truth = ground_truth_rollout[:, step]
                    
                    if hasattr(dataset, 'remove_padding'):
                        prediction_unpadded = dataset.remove_padding(prediction)
                        ground_truth_unpadded = dataset.remove_padding(ground_truth)
                    else:
                        prediction_unpadded = prediction
                        ground_truth_unpadded = ground_truth
                    
                    l1_loss = F.l1_loss(prediction_unpadded, ground_truth_unpadded)
                    vrmse = _batch_vrmse(prediction_unpadded, ground_truth_unpadded)

                    total_l1_loss += l1_loss.item()
                    total_vrmse += vrmse.item()
                    steps_evaluated += 1

                current_frame = prediction
    
    mean_l1 = total_l1_loss / steps_evaluated if steps_evaluated > 0 else 0
    mean_vrmse = total_vrmse / steps_evaluated if steps_evaluated > 0 else 0
    
    print("\n" + "="*30)
    print("Evaluation Complete")
    print(f"Rollout Range: {args.rollout_start_frame} -> {args.rollout_end_frame}")
    print(f"Mean L1 Loss: {mean_l1:.8f}")
    print(f"Mean VRMSE  : {mean_vrmse:.8f}")
    print("="*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to calculate rollout errors from the start of each sequence."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model checkpoint.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the root of the dataset.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset class to use.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    
    parser.add_argument("--rollout_start_frame", type=int, required=True, help="The starting frame of the rollout to calculate error (inclusive).")
    parser.add_argument("--rollout_end_frame", type=int, required=True, help="The ending frame of the rollout to calculate error (inclusive).")

    args = parser.parse_args()
    evaluate_l1_and_vrmse(args)