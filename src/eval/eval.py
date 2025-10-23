
import torch

from .metrics import recall_at_k, mrr, ndcg


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