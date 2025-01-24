import argparse
import json
from typing import List, Dict, Union
import torch
from PIL import Image
import requests
from io import BytesIO
import base64
import os
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_random_exponential

class ICLExperiment:
    def __init__(self, model_name: str, provider: str = "huggingface"):
        self.model_name = model_name
        self.provider = provider
        
        print(f"Loading model and processor from {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully!")

    def load_image(self, image_path: str) -> Image.Image:
        """Load image from local path or URL."""
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_path)
            return img.convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            raise

    def process_input(self, text: str, image_path: str = None):
        """Process text and optional image input."""
        if image_path:
            image = self.load_image(image_path)
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            )
        return inputs.to(self.model.device)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_completion(self, prompt: Union[str, Dict], system_prompt: str = None) -> str:
        try:
            # Handle both string and dictionary inputs
            if isinstance(prompt, dict):
                text = prompt['text']
                image_path = prompt.get('image')
            else:
                text = prompt
                image_path = None

            # Combine system prompt if provided
            if system_prompt:
                text = f"{system_prompt}\n{text}"

            # Process inputs
            inputs = self.process_input(text, image_path)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode and return only the new tokens
            response = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()
                
        except Exception as e:
            print(f"Error in model inference: {str(e)}")
            raise

    def create_few_shot_prompt(self, examples: List[Dict], test_input: Dict) -> Dict:
        """Create few-shot prompt with support for image inputs."""
        prompt_text = "Here are some examples:\n\n"
        
        # Add examples
        for idx, ex in enumerate(examples):
            prompt_text += f"Example {idx + 1}:\n"
            prompt_text += f"Input: {ex['input']['text']}\n"
            prompt_text += f"Output: {ex['output']}\n\n"
        
        # Add test input
        prompt_text += f"Now, please provide the output for this input:\n{test_input['text']}"
        
        # Return prompt with the current image path
        return {
            "text": prompt_text,
            "image": test_input.get('image')
        }

    def run_experiment(self, 
                      train_examples: List[Dict],
                      test_examples: List[Dict],
                      system_prompt: str = None) -> List[Dict]:
        results = []
        for test_ex in test_examples:
            prompt = self.create_few_shot_prompt(train_examples, test_ex['input'])
            prediction = self.get_completion(prompt, system_prompt)
            results.append({
                'input': test_ex['input'],
                'true_output': test_ex['output'],
                'predicted_output': prediction
            })
        return results

def main():
    parser = argparse.ArgumentParser(description='Run in-context learning experiments')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., deepseek-ai/DeepSeek-R1-Distill-Llama-8B)')
    parser.add_argument('--train_dir', type=str, default='srcs/example',
                        help='Directory containing train and test data')
    parser.add_argument('--system_prompt', type=str, default=None,
                       help='Optional system prompt')
    parser.add_argument('--output_file', type=str, default='results.json',
                       help='Path to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run the model on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load data
    with open(os.path.join(args.train_dir, 'train.json'), 'r') as f:
        train_examples = json.load(f)
    with open(os.path.join(args.train_dir, 'test.json'), 'r') as f:
        test_examples = json.load(f)
    
    # Initialize experiment
    experiment = ICLExperiment(args.model)
    
    # Run experiment
    results = experiment.run_experiment(
        train_examples,
        test_examples,
        args.system_prompt
    )
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()