import torch
import numpy as np

from copy import deepcopy

def load_data(path):
    """Load processed data from .pt file"""
    torch.serialization.add_safe_globals([
        np._core.multiarray._reconstruct, # ignore
        np.dtype, 
        np.ndarray, 
        np.dtypes.StrDType, 
        np.dtypes.Int64DType
    ])
    data = torch.load(path, weights_only=False)
    return data


def prepare_train_data(data):
    """Prepare training data from loaded dict"""
    caption_embd = data['caption_embd']
    image_embd = data['img_embd']
    caption2img_idx = data['caption2img_idx']
    
    X = caption_embd.float()
    # Map each caption to its corresponding image embedding
    y = image_embd[caption2img_idx].float()
    
    print(f"Train data: {len(X)} captions, {len(image_embd)} images")
    return X, y


@torch.inference_mode()
def evaluate_retrieval(model, caption_embd, image_embd, gt_indices, device, k=10):
    """Evaluate retrieval performance using cosine similarity"""
    model.eval()
    
    # Translate captions to image space
    caption_embd_tensor = caption_embd.to(device)
    translated = model(caption_embd_tensor).cpu()
    
    # Normalize embeddings for cosine similarity
    translated_norm = translated / translated.norm(dim=1, keepdim=True)
    image_embd_norm = image_embd / image_embd.norm(dim=1, keepdim=True)
    
    # Compute similarity matrix
    similarity = translated_norm @ image_embd_norm.T  # (N_captions, N_images)
    
    # Get top-k predictions
    topk_indices = similarity.topk(k, dim=1).indices  # (N_captions, k)
    
    # Compute Recall@k
    recall = 0
    for i in range(len(gt_indices)):
        if gt_indices[i] in topk_indices[i]:
            recall += 1
    recall /= len(gt_indices)
    
    # Compute MRR
    mrr = 0
    for i in range(len(gt_indices)):
        ranks = (topk_indices[i] == gt_indices[i]).nonzero(as_tuple=True)
        if len(ranks[0]) > 0:
            mrr += 1.0 / (ranks[0][0].item() + 1)
    mrr /= len(gt_indices)
    
    # Compute L2 distance to ground truth
    gt_embeddings = image_embd[gt_indices]
    l2_dist = (translated - gt_embeddings).norm(dim=1).mean().item()
    
    return recall, mrr, l2_dist

@torch.inference_mode()
def generate_submission(test_data, pred_embds, output_file="submission.pt"):
    """Generate submission file"""
    
    submission = deepcopy(test_data)

    if isinstance(pred_embds, torch.Tensor):
        pred_embds = pred_embds.cpu()
    elif isinstance(pred_embds, np.ndarray):
        pred_embds = torch.from_numpy(pred_embds).float()
    else:
        raise ValueError("pred_embds must be a torch.Tensor or numpy.ndarray")
    
    assert pred_embds.shape[0] == submission['caption_embd'].shape[0], \
        "Number of predicted embeddings must match number of captions"
    
    submission["img_embd"] = pred_embds
    torch.save(submission, output_file)
    print(f"✓ Saved submission to {output_file}")
    
    return submission