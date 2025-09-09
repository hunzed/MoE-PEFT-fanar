# Arabic Fine-tuning with CultranAI-mixlora

Two approaches for Arabic language model fine-tuning:

## 1. Standard Arabic Fine-tuning
- **Script**: `arabic_finetune_standard.py`
- **Experts**: 8 generic experts (automatic specialization)
- **Routing**: top_k=2 (collaborative experts)
- **Use case**: General purpose Arabic fine-tuning

## 2. Regional Arabic Fine-tuning  
- **Script**: `arabic_finetune_regional.py`
- **Experts**: 4 strict regional experts (Gulf, Levant, North Africa, Others)
- **Routing**: top_k=1 (single expert per input)
- **Use case**: Regional specialization with strict routing

## Quick Start

### Standard Arabic Fine-tuning
```bash
# Basic training with 8 generic experts
python arabic_finetune_standard.py

# Custom parameters
python arabic_finetune_standard.py \
    --name "my_model" \
    --epochs 3 \
    --batch_size 16
```

### Regional Arabic Fine-tuning
```bash
# Basic training with 4 regional experts
python arabic_finetune_regional.py

# Custom parameters  
python arabic_finetune_regional.py \
    --name "regional_model" \
    --epochs 5 \
    --aux_loss 0.2
```

## Configuration Differences

### Standard Fine-tuning
- **Experts**: 8 generic experts
- **Top-k**: 2 (allows expert collaboration)
- **Specialization**: Automatic (learns any patterns)
- **Datasets**: Mixed Arabic datasets

### Regional Fine-tuning  
- **Experts**: 4 regional experts (Gulf, Levant, North Africa, Others)
- **Top-k**: 1 (strict single expert routing)
- **Specialization**: Forced regional specialization
- **Datasets**: Regional datasets from `datasets/regional/`

### Common Parameters
- `--epochs`: Training epochs (default: 3)
- `--batch_size`: Batch size (default: 16) 
- `--lr`: Learning rate (default: 2e-4)
- `--r`: LoRA rank (default: 16)

## Dataset Format

Both approaches use Arabic MCQ format:

```json
{
    "question": "Question in Arabic",
    "A": "Option A", "B": "Option B", 
    "C": "Option C", "D": "Option D",
    "answer": "A",
    "country": "Yemen"  // For regional datasets
}
```

### Standard Datasets
- Mixed Arabic cultural datasets
- Generic expert specialization

### Regional Datasets  
- `datasets/regional/gulf.jsonl` 
- `datasets/regional/levant.jsonl`
- `datasets/regional/north_africa.jsonl`
- `datasets/regional/others.jsonl`

## Training Output

Both scripts save models to `outputs/` directory:

```
outputs/MODEL_NAME_TIMESTAMP/
├── adapter_config.json
└── adapter_model.bin  
```

## Usage Examples

### Standard (8 generic experts)
```bash
python arabic_finetune_standard.py --epochs 3 --batch_size 16
```

### Regional (4 regional experts)  
```bash
python arabic_finetune_regional.py --epochs 5 --aux_loss 0.2
```

### Common Options
- `--config_only`: Generate config without training
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--load_4bit`: Use 4-bit quantization
