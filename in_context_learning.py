import argparse
from datetime import datetime
import json
from typing import List, Dict, Tuple
import os

import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import MllamaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv
from tqdm import tqdm
from rich import print

load_dotenv()


class ICLExperiment:
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.is_llama = 'llama' in model_name.lower()

        print(
            f"Loading model and processor from {model_name}, model type: {model_type}...")
        if self.model_type == 'llava_llama3':
            print("Use the LlavaNext model")
            self.processor = LlavaNextProcessor.from_pretrained(
                model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_TOKEN"),
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif self.model_type == 'llama3':
            self.processor = AutoProcessor.from_pretrained(
                model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_TOKEN"),
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_TOKEN"),
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

    def process_input(self, text: str, image_paths: List[str] = None):
        """Process text and multiple images input."""
        if image_paths:
            images = [self.load_image(path) for path in image_paths]

            if self.model_type == 'llama3':
                inputs = self.processor(
                    images,
                    text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(self.model.device)
            else:
                inputs = self.processor(
                    text=text,
                    images=images,
                    return_tensors="pt",
                    padding=True
                ).to(self.model.device)
        else:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

        return inputs

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_completion(self, text: str, image_paths: List[str] = None) -> str:
        try:
            print(f'============Generate Info===========\n')
            print(f'\ntext: {text}')
            print(f'\nimage_paths: {image_paths}')
            print(f'\n==========Generate Info============\n')
            inputs = self.process_input(text, image_paths)

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

            response = self.processor.decode(outputs[0])
            output = response.replace(text, '').strip()
            print(f'============Generate Output===========\n')
            print(f'output: {output}')
            print(f'\n==========Generate Output============\n')

            return output

        except Exception as e:
            print(f"Error in model inference: {str(e)}")
            raise

    def create_context_prompt(self, examples_so_far: List[Dict], current_input: Dict) -> Tuple[str, List[str]]:
        """Create prompt using previous examples and current input, return prompt and list of image paths."""
        image_paths = []

        if self.model_type in ['llama3', 'llava_llama3']:
            # First instruction example
            print(f'\nexamples_so_far: {examples_so_far}')
            print(f'\ncurrent_input: {current_input}')

            messages = []

            for idx, ex in enumerate(examples_so_far):
                userChat = dict()
                if 'image' in ex['input']:
                    userChat['role'] = 'user'
                    userChat['content'] = [
                        {'type': "text", 'text': ex['input']['text']}, {'type': "image"}]
                    image_paths.append(ex['input']['image'])
                else:
                    userChat['role'] = 'user'
                    userChat['content'] = [
                        {'type': "text", 'text': ex['input']['text']}]
                messages.append(userChat)

                assistantChat = dict()
                if 'output' in ex:
                    assistantChat['role'] = 'assistant'
                    assistantChat['content'] = [
                        {'type': "text", 'text': ex['output']}]
                messages.append(assistantChat)

            if 'image' in current_input:
                messages.append({'role': 'user', 'content': [
                                {'type': "text", 'text': current_input['text']}, {'type': "image"}]})
                image_paths.append(current_input['image'])
            else:
                messages.append({'role': 'user', 'content': [
                                {'type': "text", 'text': current_input['text']}]})

            print(f'\nmessages: {messages}')
            prompt = self.processor.apply_chat_template(
                messages, add_generator_token=True)

        else:
            prompt = " ".join([ex['input'] for ex in examples_so_far])
            prompt += f" {current_input['text']}"

            if 'image' in current_input:
                image_paths.append(current_input['image'])
        return prompt, image_paths

    def run_sequential_training(self, train_examples: List[Dict]) -> List[Dict]:
        """Run sequential in-context learning on training examples."""
        processed_examples = [train_examples[0]]  # Start with instruction

        print("Processing training examples sequentially...")
        for i in tqdm(range(1, len(train_examples))):
            current_example = train_examples[i]
            prompt, image_paths = self.create_context_prompt(
                processed_examples, current_example['input'])

            # Get completion with all images in context
            current_example['output'] = self.get_completion(
                prompt, image_paths)
            processed_examples.append(current_example)

        return processed_examples

    def run_test_inference(self, trained_examples: List[Dict], test_examples: List[Dict]) -> List[Dict]:
        """Run inference on test examples using all training examples."""
        results = []
        print("Running inference on test examples...")
        for test_ex in tqdm(test_examples):
            prompt, image_paths = self.create_context_prompt(
                trained_examples, test_ex['input'])
            prediction = self.get_completion(prompt, image_paths)
            results.append({
                'input': test_ex['input'],
                'predicted_output': prediction
            })
        return results

    def run_experiment(self, train_examples: List[Dict], test_examples: List[Dict], args: argparse) -> Dict:
        processed_train_path = os.path.join(
            args.train_dir, 'train_with_outputs.json')
        if not os.path.exists(processed_train_path):
            # Sequential training
            trained_examples = self.run_sequential_training(train_examples)
            with open(processed_train_path, 'w') as f:
                json.dump(trained_examples, f, indent=2)
            print(
                f"Saved processed training examples to {processed_train_path}")
        else:
            with open(processed_train_path, 'r') as f:
                trained_examples = json.load(f)
            print(
                f"Loaded processed training examples from {processed_train_path}")

        # Test inference
        test_results = self.run_test_inference(trained_examples, test_examples)

        return {
            'trained_examples': trained_examples,
            'test_results': test_results
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run in-context learning experiments')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., meta-llama/Llama-2-7b-chat-hf)')
    parser.add_argument('--train_dir', type=str, default='srcs/example',
                        help='Directory containing train and test data')
    parser.add_argument('--output_file', type=str, default='',
                        help='Path to save results')
    parser.add_argument('--model_type', type=str, default='llama3',
                        help='Model type (llama3, Qwen, GLM, etc.)')

    args = parser.parse_args()

    # Load data
    with open(os.path.join(args.train_dir, 'train.json'), 'r') as f:
        train_examples = json.load(f)
    with open(os.path.join(args.train_dir, 'test.json'), 'r') as f:
        test_examples = json.load(f)

    # Initialize experiment
    experiment = ICLExperiment(args.model, args.model_type)

    # Run experiment
    results = experiment.run_experiment(train_examples, test_examples, args)
    results['model'] = args.model

    # Save results
    if not args.output_file:
        args.output_file = os.path.join(
            args.train_dir, 'results', f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_results.json')
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
