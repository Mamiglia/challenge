import argparse
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

def load_text_model(model_name="sentence-transformers/roberta-large-nli-stsb-mean-tokens"):
    """Load Sentence-BERT text encoder."""
    print(f"Loading text model: {model_name}")
    return SentenceTransformer(model_name)


def load_image_model(model_name="facebook/dinov2-giant"):
    """Load DINOv2 image encoder."""
    print(f"Loading image model: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    return image_processor, model


@torch.inference_mode()
def process_images_batch(image_processor, model, image_paths, device, batch_size=32):
    """Generate image embeddings in batches."""
    print(f"Processing {len(image_paths)} images in batches...")
    model.to(device)
    model.eval()
    
    all_embeddings = []
    failed_indices = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        valid_images = []
        
        # Keep track of which original indices correspond to valid images in the batch
        valid_original_indices = []

        for j, path in enumerate(batch_paths):
            original_index = i + j
            try:
                img = Image.open(path).convert("RGB")
                valid_images.append(img)
                valid_original_indices.append(original_index)
            except Exception as e:
                print(f"Warning: Skipping image {path} due to error: {e}")
                failed_indices.append(original_index)

        if not valid_images:
            continue

        inputs = image_processor(images=valid_images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        
        # Average over patch tokens
        image_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Store embeddings based on their success
        all_embeddings.extend(image_features)
    
    if not all_embeddings:
        return np.array([]), list(range(len(image_paths)))

    return np.vstack(all_embeddings), failed_indices


def process_captions(text_model, captions, device):
    """Generate text embeddings using Sentence-BERT."""
    print("Processing captions...")
    return text_model.encode(
        captions, 
        convert_to_numpy=True, 
        show_progress_bar=True, 
        device=device
    )

def load_dataset(dataset_path):
    """
    Load dataset from a directory containing captions.txt and an Images folder.
    """
    dataset_path = Path(dataset_path)
    captions_file = dataset_path / "captions.txt"
    images_dir = dataset_path / "Images"

    if not captions_file.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Could not find 'captions.txt' or 'Images' directory in {dataset_path}")

    df = pd.read_csv(captions_file)
    
    # Ensure 'image' and 'caption' columns exist
    if 'image' not in df.columns or 'caption' not in df.columns:
        # Try with different separator if columns are not found
        df = pd.read_csv(captions_file, sep='|')
        if 'image' not in df.columns or 'caption' not in df.columns:
             # Assuming first column is image and second is caption
            df.columns = ['image', 'caption'] + df.columns[2:].tolist()


    # Group captions by image
    captions_grouped = df.groupby('image')['caption'].apply(list).reset_index()
    
    image_files = captions_grouped['image'].tolist()
    image_paths = [images_dir / fname for fname in image_files]
    captions_by_image = captions_grouped['caption'].tolist()

    return image_files, image_paths, captions_by_image, df

def create_data_file(dataset_path, output_file, device=None, args={}):
    """
    Main function to generate embeddings and save the final .pt file.
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    text_model = load_text_model()
    image_processor, image_model = load_image_model()

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    image_files, image_paths, captions_by_image, df_captions = load_dataset(dataset_path)
    
    num_images = len(image_files)
    print(f"Found {num_images} images and {len(df_captions)} total captions.")

    # Generate image embeddings
    image_embeddings, failed_indices = process_images_batch(image_processor, image_model, image_paths, device)
    
    if failed_indices:
        print(f"Warning: Failed to process {len(failed_indices)} images. They will be excluded.")
        # Create a set of failed indices for efficient lookup
        failed_indices_set = set(failed_indices)
        
        # Filter out failed images and their corresponding captions
        original_indices = [i for i in range(num_images) if i not in failed_indices_set]
        image_files = [image_files[i] for i in original_indices]
        captions_by_image = [captions_by_image[i] for i in original_indices]
        num_images = len(image_files)

    # Flatten captions and generate embeddings
    all_captions_flat = [caption for sublist in captions_by_image for caption in sublist]
    caption_embeddings = process_captions(text_model, all_captions_flat, device)

    # Create ground truth indices
    repeats = [len(caps) for caps in captions_by_image]
    gt_indices = np.arange(num_images).repeat(repeats)

    # Prepare data dictionary
    data = dict(
        caption_embd=torch.from_numpy(caption_embeddings).float(),
        captions_text=np.array(all_captions_flat),
        img_embd=torch.from_numpy(image_embeddings).float(),
        img_file=np.array(image_files),
        caption2img_idx=gt_indices
    )

    # Save to file
    print(f"Saving processed data to {output_file}")
    torch.save(data, output_file)
    print("✓ Done.")
    
    if args.create_secret_version:
        data_secret = dict(
            caption_embd=torch.from_numpy(caption_embeddings).float(),
            captions_text=np.array(all_captions_flat),
        )
        secret_output_file = str(Path(output_file).with_name(Path(output_file).stem + "_secret.pt"))
        print(f"Saving secret version to {secret_output_file}")
        torch.save(data_secret, secret_output_file)
        print("✓ Secret version saved.")
        

def main():
    parser = argparse.ArgumentParser(description="Preprocess image-caption dataset and save to a .pt file.", add_help=True)
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the dataset folder (e.g., 'data/flickr30k_challenge/train')."
    )
    parser.add_argument(
        "--output-file", '-o',
        type=str,
        default="processed_data.pt",
        help="Path to save the output .pt file."
    )
    parser.add_argument(
        "--create-secret-version",
        action='store_true',
        help="Create a secret version of the output file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu'). Autodetects if not specified."
    )
    args = parser.parse_args()

    create_data_file(args.input_folder, args.output_file, args.device, args)

if __name__ == "__main__":
    main()
