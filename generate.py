import fire
import torch

import moe_peft


def inference_callback(cur_pos, outputs):
    print(f"Position: {cur_pos}")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} output: {output[0]}")


def main(
    base_model: str,
    instruction: str,
    input: str = None,
    template: str = None,
    lora_weights: str = None,
    load_16bit: bool = True,
    load_8bit: bool = False,
    load_4bit: bool = False,
    flash_attn: bool = False,
    max_seq_len: int = 512,
    stream: bool = False,
    device: str = moe_peft.backend.default_device_name(),
):
    # Validate device parameter to prevent Arabic characters or invalid device names
    valid_devices = ["cpu", "cuda", "auto", "balanced", "balanced_low_0", "sequential"]
    if device and not any(device.startswith(d) for d in valid_devices):
        print(f"Warning: Invalid device '{device}' detected (contains non-ASCII characters). Using default device instead.")
        device = moe_peft.backend.default_device_name()
    
    # Additional check for non-ASCII characters
    try:
        device.encode('ascii')
    except UnicodeEncodeError:
        print(f"Warning: Device name contains non-ASCII characters: '{device}'. Using default device instead.")
        device = moe_peft.backend.default_device_name()
    
    print(f"Using device: {device}")

    model = moe_peft.LLMModel.from_pretrained(
        base_model,
        device=device,
        attn_impl="flash_attn" if flash_attn else "eager",
        bits=(8 if load_8bit else (4 if load_4bit else None)),
        load_dtype=torch.bfloat16 if load_16bit else torch.float32,
    )
    tokenizer = moe_peft.Tokenizer(base_model)

    if lora_weights:
        adapter_name = model.load_adapter(lora_weights)
    else:
        adapter_name = model.init_adapter(
            moe_peft.AdapterConfig(adapter_name="default")
        )

    generate_paramas = moe_peft.GenerateConfig(
        adapter_name=adapter_name,
        prompt_template=template,
        prompts=[(instruction, input)],
    )

    output = moe_peft.generate(
        model,
        tokenizer,
        [generate_paramas],
        max_gen_len=max_seq_len,
        stream_callback=inference_callback if stream else None,
    )

    for prompt in output[adapter_name]:
        print(f"\n{'='*10}\n")
        print(prompt)
        print(f"\n{'='*10}\n")


if __name__ == "__main__":
    fire.Fire(main)
