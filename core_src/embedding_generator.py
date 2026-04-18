"""
EmbeddingGenerator: Wrapper for PE-Core-L14-336 embedding model
Generates embeddings for images and text using Perception Encoder
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
import sys
from pathlib import Path

# Add perception_models to path
perception_path = Path(__file__).parent.parent / "perception_models"
sys.path.insert(0, str(perception_path))

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms


class EmbeddingGenerator:
    """
    Generates embeddings using PE-Core-L14-336 model
    Supports both image and text embeddings with 1024-dimensional output
    """

    def __init__(self,
                 model_name: str = "PE-Core-L14-336",
                 device: str = 'cuda',
                 normalize: bool = True,
                 batch_size: int = 8):
        """
        Initialize PE-Core embedding model

        Args:
            model_name: Model configuration name
            device: Device to run on
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for processing
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.normalize = normalize
        self.batch_size = batch_size
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")

        # Load CLIP model for image and text encoding
        self.model = pe.CLIP.from_config(model_name, pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize transforms
        self.image_size = self.model.image_size  # 336 for PE-Core-L14-336
        self.context_length = self.model.context_length  # 32 for PE-Core-L14-336

        self.preprocess = transforms.get_image_transform(self.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.context_length)

        print(f"Embedding model initialized. Image size: {self.image_size}, "
              f"Context length: {self.context_length}, Embedding dim: 1024")

    def encode_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
        """
        Encode a single image to embedding

        Args:
            image: Input image (PIL Image, numpy array, or torch tensor)

        Returns:
            Image embedding as numpy array (1024-dim)
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if image.shape[-1] == 3 and len(image.shape) == 3:
                    image = Image.fromarray(image[..., ::-1])  # BGR to RGB
                else:
                    image = Image.fromarray(image)
            elif isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)

            # Preprocess
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Encode
            with torch.no_grad(), torch.autocast(self.device):
                embedding = self.model.encode_image(image_tensor, normalize=self.normalize)

            return embedding.cpu().numpy()[0]

        except Exception as e:
            print(f"Error encoding image: {e}")
            return np.zeros(1024, dtype=np.float32)

    def encode_images_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Encode multiple images in batches

        Args:
            images: List of images

        Returns:
            Image embeddings as numpy array (N, 1024)
        """
        try:
            embeddings = []

            # Process in batches
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i:i + self.batch_size]

                # Preprocess batch
                batch_tensors = []
                for img in batch_images:
                    if isinstance(img, np.ndarray):
                        # Convert BGR to RGB if needed
                        if img.shape[-1] == 3 and len(img.shape) == 3:
                            img = Image.fromarray(img[..., ::-1])
                        else:
                            img = Image.fromarray(img)
                    elif isinstance(img, torch.Tensor):
                        img = transforms.ToPILImage()(img)

                    batch_tensors.append(self.preprocess(img))

                batch_tensor = torch.stack(batch_tensors).to(self.device)

                # Encode batch
                with torch.no_grad(), torch.autocast(self.device):
                    batch_embeddings = self.model.encode_image(batch_tensor, normalize=self.normalize)

                embeddings.append(batch_embeddings.cpu().numpy())

            # Concatenate all batches
            return np.vstack(embeddings)

        except Exception as e:
            print(f"Error encoding images batch: {e}")
            return np.zeros((len(images), 1024), dtype=np.float32)

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text to embedding

        Args:
            text: Input text or list of texts

        Returns:
            Text embedding(s) as numpy array
        """
        try:
            # Handle single text
            if isinstance(text, str):
                text = [text]
                single_text = True
            else:
                single_text = False

            # Tokenize
            text_tokens = self.tokenizer(text).to(self.device)

            # Encode
            with torch.no_grad(), torch.autocast(self.device):
                embeddings = self.model.encode_text(text_tokens, normalize=self.normalize)

            result = embeddings.cpu().numpy()

            return result[0] if single_text else result

        except Exception as e:
            print(f"Error encoding text: {e}")
            if isinstance(text, str):
                return np.zeros(1024, dtype=np.float32)
            else:
                return np.zeros((len(text), 1024), dtype=np.float32)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        try:
            # Normalize if not already normalized
            if not self.normalize:
                embedding1 = embedding1 / np.linalg.norm(embedding1)
                embedding2 = embedding2 / np.linalg.norm(embedding2)

            similarity = float(np.dot(embedding1, embedding2))
            return similarity

        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0

    def compute_similarity_matrix(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix

        Args:
            embeddings1: First set of embeddings (N, 1024)
            embeddings2: Second set of embeddings (M, 1024)

        Returns:
            Similarity matrix (N, M)
        """
        try:
            # Normalize if not already normalized
            if not self.normalize:
                embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
                embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

            # Compute similarity matrix
            similarity_matrix = np.dot(embeddings1, embeddings2.T)

            return similarity_matrix

        except Exception as e:
            print(f"Error computing similarity matrix: {e}")
            return np.zeros((embeddings1.shape[0], embeddings2.shape[0]), dtype=np.float32)

    def encode_image_text_similarity(self, image: Union[np.ndarray, Image.Image],
                                     texts: List[str]) -> np.ndarray:
        """
        Encode image and texts, compute similarities

        Args:
            image: Input image
            texts: List of text descriptions

        Returns:
            Similarity scores for each text
        """
        try:
            # Encode image
            image_embedding = self.encode_image(image)

            # Encode texts
            text_embeddings = self.encode_text(texts)

            # Compute similarities
            similarities = self.compute_similarity_matrix(
                image_embedding.reshape(1, -1),
                text_embeddings
            )[0]

            return similarities

        except Exception as e:
            print(f"Error computing image-text similarities: {e}")
            return np.zeros(len(texts), dtype=np.float32)

    def get_top_k_similar(self, query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         k: int = 5) -> List[int]:
        """
        Get top-k most similar candidates

        Args:
            query_embedding: Query embedding (1024-dim)
            candidate_embeddings: Candidate embeddings (N, 1024)
            k: Number of top results

        Returns:
            List of indices of top-k candidates
        """
        try:
            # Compute similarities
            similarities = self.compute_similarity_matrix(
                query_embedding.reshape(1, -1),
                candidate_embeddings
            )[0]

            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:k]

            return top_k_indices.tolist()

        except Exception as e:
            print(f"Error getting top-k similar: {e}")
            return []

    def encode_frame_from_path(self, image_path: str) -> np.ndarray:
        """
        Load and encode image from file path

        Args:
            image_path: Path to image file

        Returns:
            Image embedding
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.encode_image(image)
        except Exception as e:
            print(f"Error encoding image from path: {e}")
            return np.zeros(1024, dtype=np.float32)

    def cleanup(self):
        """Cleanup model resources"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Embedding generator cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")


class EmbeddingCache:
    """Simple in-memory cache for embeddings"""

    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        return self.cache.get(key)

    def set(self, key: str, embedding: np.ndarray):
        """Store embedding in cache"""
        self.cache[key] = embedding

    def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.cache

    def clear(self):
        """Clear all cached embeddings"""
        self.cache.clear()

    def size(self) -> int:
        """Get number of cached embeddings"""
        return len(self.cache)
