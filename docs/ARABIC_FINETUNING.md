# Arabic Fine-tuning with CultranAI-mixlora

This guide explains how to fine-tune Arabic language models using the MixLoRA framework for culture understanding tasks.

## Overview

The setup includes:
- **Model**: QCRI/Fanar-1-9B-Instruct (Arabic instruction-tuned model)
- **Dataset**: UBC-NLP/palmx_2025_subtask1_culture (Arabic culture MCQ dataset)
- **Method**: MixLoRA (Mixture of LoRA experts)
- **Task**: Multiple Choice Question answering for cultural understanding

## Quick Start

### 1. Using the Python Manager (Recommended)

```bash
# Basic training with default settings
python arabic_finetune.py

# Customize parameters
python arabic_finetune.py \
    --name "my_arabic_model" \
    --routing "mixlora" \
    --experts 8 \
    --top_k 2 \
    --epochs 3 \
    --batch_size 8 \
    --lr 2e-4

# Create config only (for review before training)
python arabic_finetune.py --config_only --config_path "configs/my_config.json"

# Use existing config
python moe_peft.py \
    --base_model "QCRI/Fanar-1-9B-Instruct" \
    --config "configs/my_config.json" \
    --dir "./outputs/my_training"
```

### 2. Using the Shell Script

```bash
# Run with default settings
./scripts/train_arabic_fanar.sh

# Add custom arguments
./scripts/train_arabic_fanar.sh --batch_size 16 --epochs 5
```

### 3. Using Pre-made Templates

```bash
python moe_peft.py \
    --base_model "QCRI/Fanar-1-9B-Instruct" \
    --config "templates/arabic_fanar_mixlora.json" \
    --dir "./outputs/arabic_training"
```

## Configuration Options

### Routing Strategies
- `mixlora`: Mixture of LoRA experts (recommended for multi-domain tasks)
- `loramoe`: LoRA Mixture of Experts
- `lora`: Standard LoRA

### Key Parameters
- `--experts`: Number of expert modules (default: 8)
- `--top_k`: Number of experts to activate (default: 2)
- `--epochs`: Training epochs (default: 3)
- `--batch_size`: Training batch size (default: 8)
- `--lr`: Learning rate (default: 2e-4)
- `--r`: LoRA rank (default: 16)

## Dataset Format

The system automatically handles the PalmX culture dataset with the following format:

```json
{
    "question": "Question text in Arabic",
    "A": "Option A",
    "B": "Option B", 
    "C": "Option C",
    "D": "Option D",
    "answer": "A"
}
```

The data is automatically formatted using Fanar's chat template:

```python
messages = [
    {"role": "system", "content": "You're a helpful assistant..."},
    {"role": "user", "content": "Question + Options"},
    {"role": "assistant", "content": "Answer"}
]
```

## Custom Datasets

To use your own dataset:

1. **HuggingFace Dataset**:
   ```bash
   python arabic_finetune.py --dataset "your-org/your-dataset"
   ```

2. **Local JSON/JSONL File**:
   ```bash
   python arabic_finetune.py --dataset "/path/to/your/data.jsonl"
   ```

3. **Create Custom Task**: Modify `moe_peft/tasks/arabic_tasks.py` to add your own task class.

## Monitoring Training

### Tokenization Output
To see exactly what gets tokenized and fed to the model:

```bash
# Preview tokenization without training
python arabic_finetune.py --preview_tokenization

# Quick tokenization test with sample data
python quick_tokenization_test.py

# Show complete training flow
python show_training_flow.py
```

The system provides detailed tokenization logs for the first few examples showing:
- Original Arabic text
- Formatted chat template 
- Token IDs and decoded tokens
- Special token positions
- Answer location in the sequence

### Logs
Training logs are saved to `logs/` directory with timestamps. Monitor progress:

```bash
tail -f logs/arabic_training_YYYYMMDD_HHMMSS.log
```

### Evaluation
The system automatically evaluates on validation split every `--evaluate_steps` (default: 100):

```bash
# Enable evaluation during training
python arabic_finetune.py --evaluate_steps 100
```

### Checkpoints
Models are saved every `--save_step` iterations to the output directory.

## Example Configurations

### Fast Prototyping
```bash
python arabic_finetune.py \
    --name "arabic_prototype" \
    --epochs 1 \
    --batch_size 4 \
    --evaluate_steps 50
```

### Production Training
```bash
python arabic_finetune.py \
    --name "arabic_production" \
    --routing "mixlora" \
    --experts 16 \
    --top_k 4 \
    --epochs 5 \
    --batch_size 16 \
    --lr 1e-4 \
    --warmup_ratio 0.1
```

### Memory-Efficient Training
```bash
python arabic_finetune.py \
    --name "arabic_efficient" \
    --batch_size 4 \
    --micro_batch_size 2 \
    --r 8 \
    --cutoff_len 256
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `--batch_size` and `--micro_batch_size`
   - Reduce `--cutoff_len`
   - Use `--load_4bit` for quantization

2. **Dataset Loading Issues**:
   - Check dataset name/path
   - Verify internet connection for HuggingFace datasets
   - Check data format matches expected structure

3. **Tokenizer Issues**:
   - Ensure model name is correct
   - Check HuggingFace access token if needed

### Performance Tips

1. **Speed Up Training**:
   - Use `--bf16` for supported hardware
   - Enable `--tf32` on Ampere GPUs
   - Increase `--batch_size` if memory allows

2. **Improve Quality**:
   - Increase `--epochs`
   - Use higher `--r` (LoRA rank)
   - Tune `--lr` (learning rate)
   - Use `--warmup_ratio 0.1`

## Output Structure

```
outputs/arabic_training_YYYYMMDD_HHMMSS/
├── arabic_fanar_mixlora/
│   ├── adapter_config.json
│   └── adapter_model.bin
└── evaluation_results.json
```

## Integration

The trained adapters can be used with the MoE-PEFT inference system:

```bash
python moe_peft.py \
    --inference \
    --base_model "QCRI/Fanar-1-9B-Instruct" \
    --config "path/to/config.json" \
    --dir "path/to/trained/adapters"
```
