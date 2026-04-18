"""
EventDescriptionGenerator: Wrapper for SmolVLM2 2.2B Instruct model
Generates natural language descriptions of battlefield events from video frames
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union, Optional
from transformers import AutoProcessor, AutoModelForImageTextToText
import cv2


class EventDescriptionGenerator:
    """
    Generates event descriptions using SmolVLM2 2.2B Instruct
    Vision-Language Model for understanding battlefield reconnaissance footage
    """

    def __init__(self,
                 model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                 device: str = 'cuda',
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 batch_size: int = 4):
        """
        Initialize SmolVLM2 model

        Args:
            model_name: HuggingFace model name
            device: Device to run on
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            batch_size: Batch size for processing
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.model_name = model_name

        print(f"Loading VLM model: {model_name}")

        try:
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

            # Determine dtype based on device
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"VLM model initialized on {self.device} with dtype {self.dtype}")

        except ImportError as e:
            if "num2words" in str(e):
                raise ImportError(
                    "SmolVLM2 requires the 'num2words' package. "
                    "Please install it with: pip install num2words"
                ) from e
            else:
                raise
        except Exception as e:
            print(f"Error loading VLM model: {e}")
            raise

    def describe_frame(self, image: Union[np.ndarray, Image.Image],
                      prompt: Optional[str] = None,
                      objects_detected: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate description for a single frame

        Args:
            image: Input image (PIL Image or numpy array)
            prompt: Optional custom prompt
            objects_detected: Optional list of detected objects to include in context

        Returns:
            Generated description text
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed (OpenCV uses BGR format)
                if image.shape[-1] == 3 and len(image.shape) == 3:
                    # Use cv2.cvtColor for proper BGR to RGB conversion
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                else:
                    image = Image.fromarray(image)

            # Build prompt
            if prompt is None:
                prompt = self._build_default_prompt(objects_detected)

            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Process
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                text=text_prompt,
                images=[image],
                return_tensors="pt"
            )

            # Move to device and convert to correct dtype
            # pixel_values and pixel_attention_mask need dtype conversion
            inputs = {
                k: v.to(self.device).to(self.dtype) if v.dtype in [torch.float32, torch.float16, torch.bfloat16]
                else v.to(self.device)
                for k, v in inputs.items()
            }

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    top_p=0.9,
                )

            # Decode
            generated_text = self.processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]

            return generated_text.strip()

        except Exception as e:
            print(f"Error generating frame description: {e}")
            import traceback
            traceback.print_exc()
            return f"Unable to generate description: {str(e)}"

    def _build_default_prompt(self, objects_detected: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build default prompt for battlefield reconnaissance"""

        base_prompt = (
            "You are analyzing battlefield reconnaissance footage captured by a drone. "
            "Describe what you see in this frame, focusing on: "
            "1) Military objects or vehicles present "
            "2) Movement or activities "
            "3) Terrain and environmental features "
            "4) Any notable events or situations. "
            "Be concise and factual."
        )

        if objects_detected and len(objects_detected) > 0:
            # Add detected objects context
            object_types = {}
            for obj in objects_detected:
                obj_class = obj.get("class", "unknown")
                object_types[obj_class] = object_types.get(obj_class, 0) + 1

            objects_str = ", ".join([f"{count} {obj}" for obj, count in object_types.items()])
            base_prompt += f"\n\nDetected objects: {objects_str}"

        return base_prompt

    def describe_segment(self, frames: List[Union[np.ndarray, Image.Image]],
                        prompt: Optional[str] = None,
                        objects_per_frame: Optional[List[List[Dict[str, Any]]]] = None,
                        use_middle_frame: bool = True) -> str:
        """
        Generate description for a video segment

        Args:
            frames: List of frames from the segment
            prompt: Optional custom prompt
            objects_per_frame: Optional list of detected objects per frame
            use_middle_frame: If True, only use middle frame (faster)

        Returns:
            Generated description text
        """
        try:
            if not frames or len(frames) == 0:
                return "No frames available"

            # Use middle frame for efficiency
            if use_middle_frame:
                middle_idx = len(frames) // 2
                frame = frames[middle_idx]
                objects = objects_per_frame[middle_idx] if objects_per_frame else None
                return self.describe_frame(frame, prompt, objects)

            # Otherwise, use first and last frames and combine descriptions
            else:
                descriptions = []

                # First frame
                first_objects = objects_per_frame[0] if objects_per_frame else None
                first_desc = self.describe_frame(frames[0], prompt, first_objects)
                descriptions.append(f"Start: {first_desc}")

                # Last frame
                last_objects = objects_per_frame[-1] if objects_per_frame else None
                last_desc = self.describe_frame(frames[-1], prompt, last_objects)
                descriptions.append(f"End: {last_desc}")

                return " | ".join(descriptions)

        except Exception as e:
            print(f"Error generating segment description: {e}")
            import traceback
            traceback.print_exc()
            return f"Unable to generate segment description: {str(e)}"

    def describe_with_context(self, image: Union[np.ndarray, Image.Image],
                            context: Dict[str, Any]) -> str:
        """
        Generate description with additional context

        Args:
            image: Input image
            context: Context dict with keys like 'objects', 'timestamp', 'location', etc.

        Returns:
            Generated description text
        """
        try:
            # Build contextual prompt
            prompt_parts = [
                "Analyze this battlefield reconnaissance frame. "
            ]

            if "timestamp" in context:
                prompt_parts.append(f"Timestamp: {context['timestamp']}. ")

            if "objects" in context and len(context["objects"]) > 0:
                object_counts = {}
                for obj in context["objects"]:
                    obj_class = obj.get("class", "unknown")
                    object_counts[obj_class] = object_counts.get(obj_class, 0) + 1

                objects_str = ", ".join([f"{count} {obj}(s)" for obj, count in object_counts.items()])
                prompt_parts.append(f"Detected: {objects_str}. ")

            prompt_parts.append(
                "Describe the scene, activities, and any significant events. Be specific and concise."
            )

            prompt = "".join(prompt_parts)

            return self.describe_frame(image, prompt, context.get("objects"))

        except Exception as e:
            print(f"Error generating contextual description: {e}")
            import traceback
            traceback.print_exc()
            return f"Unable to generate description: {str(e)}"

    def batch_describe(self, images: List[Union[np.ndarray, Image.Image]],
                      prompts: Optional[List[str]] = None) -> List[str]:
        """
        Generate descriptions for multiple images

        Args:
            images: List of images
            prompts: Optional list of prompts (one per image)

        Returns:
            List of generated descriptions
        """
        try:
            descriptions = []

            # Process in batches (note: SmolVLM2 may not support true batching, process sequentially)
            for i, image in enumerate(images):
                prompt = prompts[i] if prompts and i < len(prompts) else None
                description = self.describe_frame(image, prompt)
                descriptions.append(description)

            return descriptions

        except Exception as e:
            print(f"Error in batch description: {e}")
            import traceback
            traceback.print_exc()
            return [f"Unable to generate description: {str(e)}"] * len(images)

    def summarize_descriptions(self, descriptions: List[str]) -> str:
        """
        Create a summary from multiple descriptions (simple concatenation)

        Args:
            descriptions: List of descriptions

        Returns:
            Summary text
        """
        try:
            if not descriptions:
                return "No descriptions available"

            # Simple approach: join with separators
            # For more sophisticated summarization, could use the VLM with a summarization prompt
            summary = " → ".join(descriptions)
            return summary

        except Exception as e:
            print(f"Error summarizing descriptions: {e}")
            return "Unable to create summary"

    def cleanup(self):
        """Cleanup model resources"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Event description generator cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")
