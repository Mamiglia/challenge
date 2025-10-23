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
def evaluate_predictions(pred_embeddings, gt_indices, test_image_emb, k=5):
    """
    Evaluate predictions on the entire dataset at once.
    
    Args:
        pred_embeddings: (N, 768) predicted image embeddings
        gt_indices: (N,) ground truth indices
        test_image_emb: (M, 768) all test image embeddings
        k: top-k retrieval
    
    Returns:
        dict with recall@k and mrr
    """
    n_samples = len(pred_embeddings)
    
    # Compute cosine similarity
    print(f"\nComputing similarity for {n_samples} predictions against {len(test_image_emb)} test images...")
    
    # Normalize embeddings for cosine similarity
    pred_norm = pred_embeddings / np.linalg.norm(pred_embeddings, axis=1, keepdims=True)
    test_norm = test_image_emb / np.linalg.norm(test_image_emb, axis=1, keepdims=True)
    
    # Compute similarity matrix (N x M)
    similarity = pred_norm @ test_norm.T
    
    # Get top-k indices for each prediction
    retrieved_indices = np.argsort(-similarity, axis=1)[:, :k]
    
    # Compute metrics
    all_recalls = []
    all_mrrs = []
    
    for i, gt_idx in enumerate(tqdm(gt_indices)):
        # Recall@k
        recall = 1 if gt_idx in retrieved_indices[i] else 0
        all_recalls.append(recall)
        
        # MRR
        mrr = 0.0
        for rank, idx in enumerate(retrieved_indices[i]):
            if idx == gt_idx:
                mrr = 1.0 / (rank + 1)
                break
        all_mrrs.append(mrr)
    
    return {
        f'recall@{k}': np.mean(all_recalls),
        'mrr': np.mean(all_mrrs),
        'n_samples': n_samples
    }

# %% [markdown]
# ## 6. Run Evaluation

# %%
results = {}
for k in K_VALUES:
    metrics = evaluate_predictions(pred_embeddings, gt_indices, image_embeddings, k=k)
    results.update(metrics) 

# %%
# Display results
results_df = pd.DataFrame([results])
print("\nEvaluation Results:")
print(results_df)