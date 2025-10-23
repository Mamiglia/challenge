import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def visualize_retrieval(pred_embeddings, text_embedding, gt_index, image_files, caption_text, 
                       image_embeddings, k=5, dataset_path="data/train"):
    """
    Visualize a single retrieval example.
    
    Args:
        translator: Translator module
        text_embedding: (384,) single text embedding
        gt_index: ground truth image index
        image_files: list of image filenames
        caption_text: the caption text
        image_embeddings: all image embeddings
        k: number of results to show
        dataset_path: path to dataset
    """    
    # Search using cosine similarity
    similarities = np.dot(image_embeddings, pred_embeddings.T).squeeze()
    retrieved_indices = np.argsort(-similarities)[:k]
    distances = -similarities[retrieved_indices]
    
    # Get ground truth image name
    gt_image_name = image_files[gt_index]
    
    # Check if ground truth is in top-k
    gt_in_topk = gt_index in retrieved_indices
    gt_rank = None
    if gt_in_topk:
        gt_rank = np.where(retrieved_indices == gt_index)[0][0] + 1
    
    # Display
    fig, axes = plt.subplots(1, k + 1, figsize=(20, 4))
    
    
    # Find the correct image path
    img_path = Path(dataset_path) / "train" / "Images" / gt_image_name

    try:
        img = Image.open(img_path)
        axes[0].imshow(img)
        axes[0].set_title(f"Ground Truth\n{gt_image_name[:20]}...", fontsize=10, color='green')
        axes[0].axis('off')
    except Exception as e:
        axes[0].text(0.5, 0.5, "Image not found", ha='center', va='center')
        axes[0].axis('off')
    
    # Retrieved images
    for i, idx in enumerate(retrieved_indices):
        retrieved_name = image_files[idx].replace('train_', '').replace('test_', '')
        
        # Find the correct image path
        img_path = Path(dataset_path) / "train" / "Images" / retrieved_name 
    
        
        try:
            img = Image.open(img_path)
            axes[i + 1].imshow(img)
            
            # Highlight if this is the ground truth
            color = 'green' if idx == gt_index else 'black'
            weight = 'bold' if idx == gt_index else 'normal'
            
            title = f"Rank {i+1}\nDist: {distances[i]:.2f}"
            if idx == gt_index:
                title += "\n✓ CORRECT"
            
            axes[i + 1].set_title(title, fontsize=10, color=color, weight=weight)
            axes[i + 1].axis('off')
        except Exception as e:
            axes[i + 1].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[i + 1].axis('off')
    
    status = f"✓ Found at rank {gt_rank}" if gt_in_topk else "✗ Not in top-5"
    plt.suptitle(f"Input: '{caption_text}'", fontsize=14, weight='bold')
    # plt.suptitle(f"Text-to-Image Retrieval - {status}", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()
    
    return gt_in_topk, gt_rank