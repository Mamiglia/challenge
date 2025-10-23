# %% [markdown]
# # Kaggle Evaluation Notebook
# 
# This notebook evaluates predictions against the secret test set.
# 
# **For instructors/Kaggle backend only!**
#
# Input: `.pt` file with dictionary containing predictions
# Output: Recall@k and MRR scores for mini-batches

# %% [markdown]
# ## 1. Setup and Imports

# %%
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# %%
# Configuration
DATASET_PATH = Path("data/test_full")  # Path to secret test data
GT_PATH = DATASET_PATH / "test_data.pt"  # Ground truth captions
PREDICTIONS_FILE = "submission.pt"  # Student submission
K_VALUES = [1, 5, 10, 20]

# %% [markdown]
# ## 2. Load Secret Test Data
#
# **This is NOT given to students!**

# %%
def load_processed_data(data_path):
    """Load pre-processed data."""
    print(f"Loading pre-processed data from {data_path}...")
    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct, np.dtype, np.ndarray, np.dtypes.StrDType, np.dtypes.Int64DType])

    data = torch.load(data_path, weights_only=False)

    caption_embeddings = data['caption_embd'].numpy() if 'caption_embd' in data else None
    captions_text = data['captions_text'] if 'captions_text' in data else None
    image_embeddings = data['img_embd'].numpy() if 'img_embd' in data else None
    gt_indices = data['caption2img_idx'] if 'caption2img_idx' in data else None
    image_files = data['img_file'] if 'img_file' in data else None

    print(f"✓ Loaded {len(image_embeddings) if image_embeddings is not None else 0} images and {len(caption_embeddings) if caption_embeddings is not None else 0} captions.")

    return caption_embeddings, image_embeddings, gt_indices, image_files, captions_text

# %%
# Load train data
caption_embeddings, image_embeddings, gt_indices, image_files, captions_text = load_processed_data(
    GT_PATH
)

_, pred_embeddings, _, _, _ = load_processed_data(
    PREDICTIONS_FILE
)

# %% [markdown]
# ## 5. Evaluation Metrics

# %%

from src.eval.metrics import recall_at_k, mrr, ndcg

@torch.inference_mode()
def evaluate_retrieval(translated_embd, image_embd, gt_indices, max_indices = 100):
    """Evaluate retrieval performance using cosine similarity"""
    # Compute similarity matrix
    if isinstance(translated_embd, np.ndarray):
        translated_embd = torch.from_numpy(translated_embd).float()
    if isinstance(image_embd, np.ndarray):
        image_embd = torch.from_numpy(image_embd).float()
    similarity = translated_embd @ image_embd.T  # (N_captions, N_images)
    
    # Get top-k predictions
    sorted_indices = similarity.topk(k=max_indices, dim=1, sorted=True).indices  # (N_captions, N_images)

    metrics = {
        'mrr': mrr,
        'ndcg': ndcg,
        'recall_at_1': lambda preds, gt: recall_at_k(preds, gt, 1),
        'recall_at_3': lambda preds, gt: recall_at_k(preds, gt, 3),
        'recall_at_5': lambda preds, gt: recall_at_k(preds, gt, 5),
        'recall_at_10': lambda preds, gt: recall_at_k(preds, gt, 10),
        'recall_at_50': lambda preds, gt: recall_at_k(preds, gt, 50),
    }
    
    results = {
        name: func(sorted_indices, gt_indices)
        for name, func in metrics.items()
    }
  
    # Compute L2 distance to ground truth
    gt_embeddings = image_embd[gt_indices]
    results['l2_dist'] = (translated_embd - gt_embeddings).norm(dim=1).mean().item()
    
    return results

# %% [markdown]
# ## 6. Run Evaluation

# %%
results = evaluate_retrieval(pred_embeddings, image_embeddings, gt_indices)

# %%
# Display results
results_df = pd.DataFrame([results])
print("\nEvaluation Results:")
print(results_df)