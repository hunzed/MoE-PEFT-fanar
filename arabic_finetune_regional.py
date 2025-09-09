#!/usr/bin/env python3
"""
Strict Regional Expert Training Script
Forces each expert to specialize on a specific region with strict routing
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any


class StrictRegionalExpertTrainer:
    """Trains strict regional experts - each expert forced to specialize on one region"""
    
    def __init__(self):
        self.base_models = {
            "fanar": "QCRI/Fanar-1-9B-Instruct",
            "llama2": "meta-llama/Llama-2-7b-hf", 
            "llama3": "meta-llama/Meta-Llama-3-8B",
        }
        
        # Fixed regional datasets
        self.regional_datasets = {
            "gulf": "./datasets/regional/gulf.jsonl",
            "levant": "./datasets/regional/levant.jsonl", 
            "north_africa": "./datasets/regional/north_africa.jsonl",
            "others": "./datasets/regional/others.jsonl"
        }
        
        self.regions = list(self.regional_datasets.keys())
        
    def create_strict_regional_config(
        self,
        name: str = "strict_regional_experts",
        model: str = "fanar",
        routing_strategy: str = "mixlora",
        epochs: int = 3,
        batch_size: int = 16,
        micro_batch_size: int = 4,
        lr: float = 2e-4,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create configuration for STRICT regional expert training
        
        Key differences from generic MoE:
        1. Exactly 4 experts for 4 regions
        2. top_k=1 for strict regional routing (no expert mixing)
        3. Higher auxiliary loss to force specialization
        4. All regional datasets combined for router to learn patterns
        """
        
        # Combine all regional datasets
        all_regional_data = ";".join(self.regional_datasets.values())
        
        config = {
            "cutoff_len": kwargs.get("cutoff_len", 512),
            "save_step": kwargs.get("save_step", 1000),
            "train_lora_candidate_num": 4,  # EXACTLY 4 candidates = 4 regions
            "train_lora_simultaneously_num": 4,  # Train all 4 simultaneously
            "train_strategy": "optim",
            "lora": [
                {
                    "name": name,
                    "task_name": "palmx_culture",  # Same task as normal Arabic finetuning
                    "optim": "adamw",
                    "scheduler_type": kwargs.get("scheduler", "cosine"),
                    "warmup_steps": kwargs.get("warmup_steps", 200),
                    "lr": lr,
                    "batch_size": batch_size,
                    "micro_batch_size": micro_batch_size,
                    "evaluate_batch_size": kwargs.get("eval_batch_size", 16),
                    "num_epochs": epochs,
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
                    
                    # STRICT REGIONAL CONFIGURATION
                    "routing_strategy": routing_strategy,
                    "num_experts": 4,  # EXACTLY 4 experts
                    "top_k": 1,        # ONLY 1 expert active = strict regional routing
                    
                    # Higher auxiliary loss forces specialization
                    "router_aux_loss_coef": kwargs.get("aux_loss", 0.1),  # Higher than default 0.001
                    
                    # Jitter noise helps exploration during training
                    "jitter_noise": kwargs.get("jitter", 0.1),  # Higher than default
                    
                    # Don't group by length - we want mixed regional data
                    "group_by_length": False,
                    
                    # All regional data combined - router learns regional patterns
                    "data": all_regional_data
                }
            ]
        }
        
        # Add routing-specific parameters for strict specialization
        if routing_strategy == "mixlora-dynamic":
            config["lora"][0]["top_p"] = 0.1  # Very low top_p for strict selection
            config["lora"][0]["temperature"] = 0.1  # Low temperature for deterministic routing
        elif routing_strategy == "mixlora-switch":
            config["lora"][0]["expert_capacity"] = kwargs.get("expert_capacity", 64)
            config["lora"][0]["router_z_loss_coef"] = kwargs.get("z_loss", 0.01)
            config["lora"][0]["ffn_dropout"] = kwargs.get("ffn_dropout", 0.1)
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """Save config file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ Config saved: {config_path}")
        return config_path


def main():
    parser = argparse.ArgumentParser(description="Train strict regional experts")
    
    # Model and naming
    parser.add_argument("--model", default="fanar", choices=["fanar", "llama2", "llama3"],
                       help="Base model to use")
    parser.add_argument("--name", default="strict_regional_experts", 
                       help="Training run name")
    
    # Core regional expert settings
    parser.add_argument("--routing", default="mixlora", 
                       choices=["mixlora", "mixlora-dynamic", "mixlora-switch"],
                       help="Routing strategy (mixlora recommended for regional)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    
    # Strict specialization parameters
    parser.add_argument("--aux_loss", type=float, default=0.1, 
                       help="Auxiliary loss coefficient (higher = more specialization)")
    parser.add_argument("--jitter", type=float, default=0.1,
                       help="Jitter noise (higher = more exploration)")
    
    # Output control
    parser.add_argument("--config_path", default="configs/strict_regional_experts.json",
                       help="Config file path")
    parser.add_argument("--config_only", action="store_true", 
                       help="Only create config without training")
    parser.add_argument("--output_dir", help="Output directory for trained model")
    
    # Hardware options
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    parser.add_argument("--load_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--load_8bit", action="store_true", help="Use 8-bit quantization")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = StrictRegionalExpertTrainer()
    

    
    # Create configuration
    config = trainer.create_strict_regional_config(
        name=args.name,
        model=args.model,
        routing_strategy=args.routing,
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        r=args.r,
        aux_loss=args.aux_loss,
        jitter=args.jitter
    )
    
    # Save config
    config_path = trainer.save_config(config, args.config_path)
    
    if args.config_only:
        print("✅ Config created!")
        return
    
    # Start training
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./outputs/strict_regional_{timestamp}"
    
    model_path = trainer.base_models.get(args.model, args.model)
    
    # Prepare training command
    moe_peft_path = os.path.join(os.getcwd(), "moe_peft.py")
    cmd = [
        "python", moe_peft_path,
        "--base_model", model_path,
        "--config", config_path,
        "--dir", args.output_dir
    ]
    
    # Add hardware options
    if args.bf16:
        cmd.append("--bf16")
    if args.load_4bit:
        cmd.append("--load_4bit")
    if args.load_8bit:
        cmd.append("--load_8bit")
    
    print(f"Training: {args.output_dir}")
    
    # Run training
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
