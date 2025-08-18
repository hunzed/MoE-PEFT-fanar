#!/usr/bin/env python3
"""
Arabic Fine-tuning Manager for CultranAI-mixlora
Provides an easy interface to fine-tune Arabic models with configurable parameters
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any


class ArabicFinetuneManager:
    """Manager class for Arabic fine-tuning tasks"""
    
    def __init__(self):
        self.base_models = {
            "fanar": "QCRI/Fanar-1-9B-Instruct",
        }
        
        self.datasets = {
            "palmx_culture": "UBC-NLP/palmx_2025_subtask1_culture",
        }
        
        self.routing_strategies = ["mixlora", "loramoe", "lora"]
    
    def create_config(
        self,
        name: str,
        task_name: str = "palmx_culture",
        dataset: str = "UBC-NLP/palmx_2025_subtask1_culture",
        model: str = "fanar",
        routing_strategy: str = "mixlora",
        num_experts: int = 8,
        top_k: int = 2,
        num_epochs: int = 3,
        batch_size: int = 8,
        micro_batch_size: int = 4,
        learning_rate: float = 2e-4,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        cutoff_len: int = 512,
        warmup_ratio: float = 0.1,
        save_step: int = 500,
        evaluate_steps: int = 100,
    ) -> Dict[str, Any]:
        """Create a configuration dictionary for Arabic fine-tuning"""
        
        config = {
            "cutoff_len": cutoff_len,
            "save_step": save_step,
            "train_lora_candidate_num": 2,
            "train_lora_simultaneously_num": 2,
            "train_strategy": "optim",
            "lora": [
                {
                    "name": name,
                    "task_name": task_name,
                    "data": f"{dataset}:train",
                    "optim": "adamw",
                    "scheduler_type": "linear",
                    "warmup_ratio": warmup_ratio,
                    "lr": learning_rate,
                    "batch_size": batch_size,
                    "micro_batch_size": micro_batch_size,
                    "evaluate_batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "r": r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "target_modules": {
                        "q_proj": True,
                        "k_proj": True,
                        "v_proj": True,
                        "o_proj": True,
                        "gate_proj": True,
                        "down_proj": True,
                        "up_proj": True
                    },
                    "routing_strategy": routing_strategy,
                    "group_by_length": False,
                }
            ]
        }
        
        # Add expert configuration for MixLoRA and LoRAMoE
        if routing_strategy in ["mixlora", "loramoe"]:
            config["lora"][0]["num_experts"] = num_experts
            config["lora"][0]["top_k"] = top_k
        
        # Add evaluation configuration
        if evaluate_steps and evaluate_steps > 0:
            config["lora"][0]["evaluate_steps"] = evaluate_steps
            config["lora"][0]["evaluate"] = [
                {
                    "name": name,
                    "task_name": task_name,
                    "data": f"{dataset}:eval",
                    "batch_size": batch_size
                }
            ]
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to: {config_path}")
    
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
        additional_args: list = None
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
        
        print(f"Starting training...")
        print(f"Command: {' '.join(cmd)}")
        print(f"Output directory: {output_dir}")
        print(f"Log file: {log_file}")
        
        # Run training
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"Training completed successfully!")
            print(f"Results saved to: {output_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Training failed with exit code: {e.returncode}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Arabic Fine-tuning Manager")
    parser.add_argument("--name", type=str, default="arabic_fanar_mixlora", help="Adapter name")
    parser.add_argument("--task", type=str, default="palmx_culture", help="Task name")
    parser.add_argument("--dataset", type=str, default="UBC-NLP/palmx_2025_subtask1_culture", help="Dataset name")
    parser.add_argument("--model", type=str, default="fanar", choices=["fanar"], help="Base model")
    parser.add_argument("--routing", type=str, default="mixlora", choices=["mixlora", "loramoe", "lora"], help="Routing strategy")
    parser.add_argument("--experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k experts")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--cutoff_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--save_step", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--evaluate_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--config_only", action="store_true", help="Only create config file, don't run training")
    parser.add_argument("--config_path", type=str, help="Path to save/load config file")
    parser.add_argument("--use_config", type=str, help="Path to existing config file to use for training (skips config creation)")
    parser.add_argument("--output_dir", type=str, help="Output directory for training results")
    parser.add_argument("--log_file", type=str, help="Log file path")
    parser.add_argument("--device", type=str, help="Device to use for training")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bfloat16")
    parser.add_argument("--no_tf32", action="store_true", help="Disable tf32")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    
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
            print(f"Using existing config file: {config_path}")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
            
    else:
        # Create new config
        config = manager.create_config(
            name=args.name,
            task_name=args.task,
            dataset=args.dataset,
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
        print("Configuration file created successfully!")
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
    )
    
    if success:
        print("Fine-tuning completed successfully!")
    else:
        print("Fine-tuning failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
