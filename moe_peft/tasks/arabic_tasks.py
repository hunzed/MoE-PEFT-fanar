import json
import logging
from typing import Any, Dict, List, Optional

import datasets as hf_datasets
import torch
from transformers import AutoTokenizer

from ..modules import InputData
from .common import BasicTask, BasicMetric, CommonSenseTask


class ArabicMCQMetric(BasicMetric):
    """Custom metric for Arabic Multiple Choice Questions"""
    
    def __init__(self) -> None:
        super().__init__()
        self.correct_predictions = 0
        self.total_predictions = 0
        
    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        # Convert tensors to lists if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().tolist()
        if isinstance(references, torch.Tensor):
            references = references.cpu().tolist()
            
        # Count correct predictions
        for pred, ref in zip(predictions, references):
            if pred == ref:
                self.correct_predictions += 1
            self.total_predictions += 1
    
    def compute(self) -> Dict[str, Any]:
        if self.total_predictions == 0:
            return {"accuracy": 0.0}
        
        accuracy = self.correct_predictions / self.total_predictions
        return {
            "accuracy": accuracy,
            "correct": self.correct_predictions,
            "total": self.total_predictions
        }


class ArabicMCQTask(CommonSenseTask):
    """Arabic Multiple Choice Question Task for culture understanding"""
    
    def __init__(
        self, 
        dataset_name: str = "UBC-NLP/palmx_2025_subtask1_culture",
        model_name: str = "QCRI/Fanar-1-9B-Instruct",
        max_length: int = 512,
        split: str = "train"
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_length = max_length
        self.split = split
        self.tokenizer = None
        # Override the label_dtype_ to use int (CommonSenseTask sets it to None)
        self.label_dtype_ = torch.int
        
    def _get_tokenizer(self):
        """Initialize tokenizer if not already done"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def format_mcq(self, row: Dict[str, Any]) -> Dict[str, str]:
        """Format MCQ data using chat template"""
        tokenizer = self._get_tokenizer()
        
        messages = [
            {
                "role": "system", 
                "content": "You're a helpful assistant that answers multiple-choice questions accurately. Choose the best answer based only on the given question and options."
            },
            {
                "role": "user", 
                "content": f"{row['question']}\n\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
            },
            {
                "role": "assistant", 
                "content": row["answer"]
            },
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            max_length=self.max_length,
            truncation=True
        )
        
        return {"text": text}
    
    def label_list(self) -> List[str]:
        """Return the list of possible answer labels for MCQ"""
        return ["A", "B", "C", "D"]
    
    @property
    def peft_task_type(self) -> str:
        return "CAUSAL_LM"
    
    def loading_data(
        self, 
        is_train: bool = True, 
        path: Optional[str] = None
    ) -> List[InputData]:
        """Load and format Arabic MCQ data"""
        
        # Handle multiple datasets separated by semicolons
        if path is not None and ";" in path:
            logging.info(f"Loading multiple datasets: {path}")
            dataset_paths = [p.strip() for p in path.split(";")]
            all_data: List[InputData] = []
            
            for dataset_path in dataset_paths:
                if dataset_path:  # Skip empty paths
                    logging.info(f"Loading dataset: {dataset_path}")
                    dataset_data = self._load_single_dataset(dataset_path, is_train)
                    all_data.extend(dataset_data)
                    logging.info(f"Loaded {len(dataset_data)} examples from {dataset_path}")
            
            logging.info(f"Total examples loaded from all datasets: {len(all_data)}")
            return all_data
        else:
            # Single dataset
            dataset_path = path if path is not None else self.dataset_name
            return self._load_single_dataset(dataset_path, is_train)
    
    def _load_single_dataset(self, dataset_path: str, is_train: bool = True) -> List[InputData]:
        """Load data from a single dataset"""
        
        # Use provided path or default dataset
        if ":" in dataset_path:
            # Handle dataset_name:split format
            dataset_parts = dataset_path.split(":")
            dataset_name = dataset_parts[0]
            split = dataset_parts[1] if len(dataset_parts) > 1 else self.split
        else:
            dataset_name = dataset_path
            split = self.split if hasattr(self, 'split') else ("train" if is_train else "dev")
        
        logging.info(f"Loading Arabic MCQ data from {dataset_name}, split: {split}")
        
        try:
            # Load the dataset
            if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
                data = hf_datasets.load_dataset("json", data_files=dataset_name)
                dataset = data[split] if split in data else data["train"]
            else:
                data = hf_datasets.load_dataset(dataset_name)
                dataset = data[split] if split in data else data["train"]
                
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            raise e
        
        ret: List[InputData] = []
        
        logging.info(f"Processing {len(dataset)} examples from {dataset_name}")
        
        for idx, row in enumerate(dataset):
            try:
                # Format the MCQ using chat template
                formatted_data = self.format_mcq(row)
                formatted_text = formatted_data["text"]
                
                # Create InputData object
                data_point = InputData(
                    inputs=formatted_text,
                    tokens=None,  # Will be tokenized later by the training pipeline
                    labels=None   # Labels will be set to tokens.copy() in dispatcher if None
                )
                
                ret.append(data_point)
                
                # Log progress for large datasets
                if idx % 1000 == 0 and idx > 0:
                    logging.info(f"Processed {idx}/{len(dataset)} examples")
                    
            except Exception as e:
                logging.warning(f"Failed to process example {idx}: {e}")
                continue
        
        logging.info(f"Successfully processed {len(ret)} examples from {dataset_name}")
        return ret
    
    def loading_metric(self) -> BasicMetric:
        """Return appropriate metric for this task"""
        return ArabicMCQMetric()
    
    def init_kwargs(self) -> Dict:
        """Return initialization kwargs for the model"""
        return {
            "task_type": "CAUSAL_LM",
            "model_name": self.model_name,
            "max_length": self.max_length
        }


def update_task_dict(task_dict: Dict[str, BasicTask]):
    """Update the global task dictionary with Arabic tasks"""
    
    # Default Arabic MCQ task with Fanar model
    task_dict["arabic_mcq"] = ArabicMCQTask()
    
    # Specific PalmX culture task
    task_dict["palmx_culture"] = ArabicMCQTask(
        dataset_name="UBC-NLP/palmx_2025_subtask1_culture",
        model_name="QCRI/Fanar-1-9B-Instruct",
        max_length=512,
        split="train"
    )
    
    # Configurable task variants
    task_dict["arabic_mcq_fanar"] = ArabicMCQTask(
        model_name="QCRI/Fanar-1-9B-Instruct"
    )
    
    logging.info("Added Arabic MCQ tasks: arabic_mcq, palmx_culture, arabic_mcq_fanar")
