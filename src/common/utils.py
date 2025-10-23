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