#!/usr/bin/env python3
"""
Arabic Fine-tuning Manager for CultranAI-mixlora
Provides an easy interface to fine-tune Arabic models with configurable parameters
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, Optional


class ArabicFinetuneManager:
    """Manager class for Arabic fine-tuning tasks"""
    
    def __init__(self):
        self.base_models = {
            "fanar": "QCRI/Fanar-1-9B-Instruct",
        }
        
        self.datasets = {
            "palmx_culture": "UBC-NLP/palmx_2025_subtask1_culture",
            "palmx_ext": "./datasets/palmX-ext.jsonl",
            "palm_train": "./datasets/palm_train.jsonl",
            "all_datasets": "UBC-NLP/palmx_2025_subtask1_culture:train;./datasets/palmX-ext.jsonl;./datasets/palm_train.jsonl",
        }
        
        self.routing_strategies = ["mixlora", "loramoe", "lora", "mixlora_dynamic", "mola"]
        
        # Template mapping for routing strategies and models
        self.template_mapping = {
            # Basic templates (default for any model)
            "mixlora": "mixlora.json",
            "loramoe": "loramoe.json", 
            "lora": "lora.json",
            "mixlora_dynamic": "mixlora_dynamic.json",
            "mola": "mola.json",
            
            # Model-specific templates
            "mixlora_glm": "mixlora_glm.json",
            "mixlora_phi": "mixlora_phi.json", 
            "mixlora_phi3": "mixlora_phi3.json",
            "mixlora_dynamic_glm": "mixlora_dynamic_glm.json",
            "mixlora_dynamic_phi": "mixlora_dynamic_phi.json",
            "mixlora_dynamic_phi3": "mixlora_dynamic_phi3.json",
            "loramoe_glm": "loramoe_glm.json",
            "loramoe_phi": "loramoe_phi.json",
            "loramoe_phi3": "loramoe_phi3.json",
            "lora_glm": "lora_glm.json",
            "lora_phi": "lora_phi.json",
            "lora_phi3": "lora_phi3.json",
            "mola_glm": "mola_glm.json", 
            "mola_phi": "mola_phi.json",
            "mola_phi3": "mola_phi3.json",
            
            # Special Arabic template
            "arabic_fanar_mixlora": "arabic_fanar_mixlora.json"
        }
    
    def get_template_key(self, routing_strategy: str, model: str = "fanar") -> str:
        """Get the best template key based on routing strategy and model"""
        

        
        # Try model-specific template first
        model_specific_key = f"{routing_strategy}_{model}"
        if model_specific_key in self.template_mapping:
            return model_specific_key
        
        # Fall back to basic routing strategy template
        if routing_strategy in self.template_mapping:
            return routing_strategy
        
        raise ValueError(f"No template found for routing strategy: {routing_strategy}")
    
    def load_template(self, routing_strategy: str, model: str = "fanar") -> Dict[str, Any]:
        """Load the appropriate template configuration"""
        template_key = self.get_template_key(routing_strategy, model)
        template_file = self.template_mapping[template_key]
        
        template_path = os.path.join("templates", template_file)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_config = json.load(f)
        
        return template_config
    
    def create_config(
        self,
        name: str,
        task_name: str = "palmx_culture",
        dataset: str = None,  # Changed to None to allow auto-detection
        model: str = "fanar",
        routing_strategy: str = "mixlora",
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None,
        cutoff_len: Optional[int] = None,
        warmup_ratio: Optional[float] = None,
        save_step: Optional[int] = None,
        evaluate_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a configuration dictionary for Arabic fine-tuning using template as base"""
        
        # Auto-detect dataset if not provided
        if dataset is None:
            if task_name in self.datasets:
                dataset = self.datasets[task_name]
            else:
                # Default fallback
                dataset = self.datasets["palmx_culture"]
        
        # Load the base template (this gives us all the defaults)
        config = self.load_template(routing_strategy, model)
        
        # Override global settings only if explicitly provided
        if cutoff_len is not None:
            config["cutoff_len"] = cutoff_len
        if save_step is not None:
            config["save_step"] = save_step
        
        # Override LoRA-specific settings
        lora_config = config["lora"][0]
        
        # Always override these (required for Arabic fine-tuning)
        lora_config["name"] = name
        lora_config["task_name"] = task_name
        
        # Handle dataset path - multiple datasets use semicolon separation
        if isinstance(dataset, list):
            # Convert list to semicolon-separated string (shouldn't happen with new implementation but kept for safety)
            lora_config["data"] = ";".join(dataset)
        elif ";" in dataset:
            # Multiple datasets already in proper format
            lora_config["data"] = dataset
        elif dataset.startswith("./") or os.path.isabs(dataset):
            # Local file - no need for :train suffix
            lora_config["data"] = dataset
        else:
            # HuggingFace dataset - add :train suffix
            lora_config["data"] = f"{dataset}:train"
        
        # Override only explicitly provided parameters (leave template defaults otherwise)
        if num_epochs is not None:
            lora_config["num_epochs"] = num_epochs
        if batch_size is not None:
            lora_config["batch_size"] = batch_size
            lora_config["evaluate_batch_size"] = batch_size  # Keep them in sync
        if micro_batch_size is not None:
            lora_config["micro_batch_size"] = micro_batch_size
        if learning_rate is not None:
            lora_config["lr"] = learning_rate
        if r is not None:
            lora_config["r"] = r
        if lora_alpha is not None:
            lora_config["lora_alpha"] = lora_alpha
        if lora_dropout is not None:
            lora_config["lora_dropout"] = lora_dropout
        if warmup_ratio is not None:
            # Use warmup_ratio if provided, otherwise keep template's warmup settings
            lora_config["warmup_ratio"] = warmup_ratio
            # Remove warmup_steps if warmup_ratio is set
            if "warmup_steps" in lora_config:
                del lora_config["warmup_steps"]
            # Set scheduler_type to linear for warmup_ratio
            lora_config["scheduler_type"] = "linear"
        
        # Handle expert configuration for MixLoRA and LoRAMoE (only override if provided)
        if routing_strategy in ["mixlora", "loramoe", "mixlora_dynamic", "mola"]:
            if num_experts is not None:
                lora_config["num_experts"] = num_experts
            if top_k is not None:
                lora_config["top_k"] = top_k
        
        # Handle evaluation configuration
        if evaluate_steps is not None and evaluate_steps > 0:
            lora_config["evaluate_steps"] = evaluate_steps
            
            # For evaluation, always use the original PalmX dataset with :eval split
            # Local datasets and combined datasets use original PalmX for evaluation
            if isinstance(dataset, list) or ";" in dataset or dataset.startswith("./") or os.path.isabs(dataset):
                # Multiple datasets or local files - use original PalmX dataset for evaluation
                eval_data = f"{self.datasets['palmx_culture']}:eval"
            else:
                # Single HuggingFace dataset - add :eval suffix
                eval_data = f"{dataset}:eval"
            
            lora_config["evaluate"] = [
                {
                    "name": name,
                    "task_name": task_name,
                    "data": eval_data,
                    "batch_size": lora_config["batch_size"]
                }
            ]
        elif evaluate_steps == 0:
            # Remove evaluation if explicitly set to 0
            if "evaluate_steps" in lora_config:
                del lora_config["evaluate_steps"]
            if "evaluate" in lora_config:
                del lora_config["evaluate"]
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_path: str, template_info: str = ""):
        """Save configuration to file"""
        config_dir = os.path.dirname(config_path)
        if config_dir:  # Only create directory if there is one
            os.makedirs(config_dir, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
    
    def run_training(
        self,
        config_path: str,
        base_model: str,
        output_dir: str = None,
        log_file: str = None,
        use_bf16: bool = True,
        use_tf32: bool = True,
        device: str = None,
        verbose: bool = True,
        additional_args: list = None,
        wandb_project: str = None,
        wandb_name: str = None,
        wandb_tags: list = None,
        no_wandb: bool = False
    ):
        """Run the training process"""
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./outputs/arabic_training_{timestamp}"
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"./logs/arabic_training_{timestamp}.log"
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Copy config file to output directory as training_config.json
        config_output_path = os.path.join(output_dir, "training_config.json")
        shutil.copy2(config_path, config_output_path)
        
        # Set up wandb environment variables if provided
        if no_wandb:
            os.environ["WANDB_DISABLED"] = "true"
        elif wandb_project and not os.getenv("WANDB_DISABLED"):
            os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_name:
            os.environ["WANDB_NAME"] = wandb_name
        if wandb_tags:
            os.environ["WANDB_TAGS"] = ",".join(wandb_tags)
        
        # Build command
        cmd = [
            sys.executable, "moe_peft.py",
            "--base_model", base_model,
            "--config", config_path,
            "--dir", output_dir,
            "--log_file", log_file,
        ]
        
        if use_bf16:
            cmd.append("--bf16")
        if use_tf32:
            cmd.append("--tf32")
        if device:
            cmd.extend(["--device", device])
        if verbose:
            cmd.append("--verbose")
        
        if additional_args:
            cmd.extend(additional_args)
        
        # Run training
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Training failed with exit code: {e.returncode}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Arabic Fine-tuning Manager")
    parser.add_argument("--name", type=str, default="arabic_fanar_mixlora", help="Adapter name")
    parser.add_argument("--data", type=str, default="palmx_culture", choices=["palmx_culture", "palmx_ext", "palm_train", "all_datasets"], help="Data source (palmx_culture: HuggingFace dataset with train/eval splits, palmx_ext: Local UAE culture dataset (train only), palm_train: Local general Arabic dataset (train only), all_datasets: All three datasets combined for training)")
    parser.add_argument("--dataset", type=str, help="Custom dataset name/path (overrides auto-selection from data)")
    parser.add_argument("--model", type=str, default="fanar", choices=["fanar"], help="Base model")
    parser.add_argument("--routing", type=str, default="mixlora", choices=["mixlora", "loramoe", "lora", "mixlora_dynamic", "mola"], help="Routing strategy")
    parser.add_argument("--experts", type=int, help="Number of experts (uses template default if not specified)")
    parser.add_argument("--top_k", type=int, help="Top-k experts (uses template default if not specified)")
    parser.add_argument("--epochs", type=int, help="Number of epochs (uses template default if not specified)")
    parser.add_argument("--batch_size", type=int, help="Batch size (uses template default if not specified)")
    parser.add_argument("--micro_batch_size", type=int, help="Micro batch size (uses template default if not specified)")
    parser.add_argument("--lr", type=float, help="Learning rate (uses template default if not specified)")
    parser.add_argument("--r", type=int, help="LoRA rank (uses template default if not specified)")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha (uses template default if not specified)")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout (uses template default if not specified)")
    parser.add_argument("--cutoff_len", type=int, help="Maximum sequence length (uses template default if not specified)")
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio (uses template default if not specified)")
    parser.add_argument("--save_step", type=int, help="Save checkpoint every N steps (uses template default if not specified)")
    parser.add_argument("--evaluate_steps", type=int, help="Evaluate every N steps (uses template default if not specified, set to 0 to disable)")
    parser.add_argument("--config_only", action="store_true", help="Only create config file, don't run training")
    parser.add_argument("--config_path", type=str, help="Path to save/load config file")
    parser.add_argument("--use_config", type=str, help="Path to existing config file to use for training (skips config creation)")
    parser.add_argument("--output_dir", type=str, help="Output directory for training results")
    parser.add_argument("--log_file", type=str, help="Log file path")
    parser.add_argument("--device", type=str, help="Device to use for training")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bfloat16")
    parser.add_argument("--no_tf32", action="store_true", help="Disable tf32")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, help="Wandb run name")
    parser.add_argument("--wandb_tags", nargs="+", help="Wandb tags (space-separated)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    manager = ArabicFinetuneManager()
    
    # Check if using existing config file
    if args.use_config:
        config_path = args.use_config
        
        # Verify config file exists
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
        # Load and validate config
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
            
    else:
        # Create new config
        # Determine dataset - use custom dataset if provided, otherwise auto-select from data choice
        dataset_to_use = args.dataset
        if dataset_to_use is None:
            if args.data in manager.datasets:
                dataset_to_use = manager.datasets[args.data]
            else:
                dataset_to_use = manager.datasets["palmx_culture"]
        
        config = manager.create_config(
            name=args.name,
            task_name="palmx_culture",  # Task is always palmx_culture
            dataset=dataset_to_use,
            model=args.model,
            routing_strategy=args.routing,
            num_experts=args.experts,
            top_k=args.top_k,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            learning_rate=args.lr,
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            cutoff_len=args.cutoff_len,
            warmup_ratio=args.warmup_ratio,
            save_step=args.save_step,
            evaluate_steps=args.evaluate_steps,
        )
        
        # Save config
        if args.config_path:
            config_path = args.config_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_path = f"configs/arabic_{args.routing}_{timestamp}.json"
        
        manager.save_config(config, config_path)
    
    if args.config_only:
        return
    
    # Get base model
    base_model = manager.base_models.get(args.model, args.model)
    
    # Run training
    success = manager.run_training(
        config_path=config_path,
        base_model=base_model,
        output_dir=args.output_dir,
        log_file=args.log_file,
        use_bf16=not args.no_bf16,
        use_tf32=not args.no_tf32,
        device=args.device,
        verbose=not args.quiet,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        no_wandb=args.no_wandb,
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
