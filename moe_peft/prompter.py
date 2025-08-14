import json
import logging
import os.path as osp
from typing import Dict, Optional, Union

prompt_templates = {
    "moe_peft": {
        "description": "Default Prompt Template Provided by MoE-PEFT",
        "prompt_input": "{input}\n الجواب: :\n",
        # "prompt_no_input": "### Instruction:\n{instruction}\n\n### Output:\n",
        "response_split": "الجواب: :",
    },
    "alpaca": {
        "description": "Template used by Alpaca-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, "
        + "paired with an input that provides further context. "
        + "Write a response that appropriately completes the request.\n\n"
        + "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. "
        + "Write a response that appropriately completes the request.\n\n"
        + "### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:",
    },
}


# manage templates and prompt building.
class Prompter:
    def __init__(self, template: Optional[Union[Dict, str]] = None):
        if template is None:
            self.template = prompt_templates["moe_peft"]
        elif isinstance(template, str):
            if osp.exists(template):
                with open(template) as fp:
                    self.template = json.load(fp)
            else:
                self.template = prompt_templates[template]
        else:
            self.template = template

        logging.info(f"Using prompt template: {self.template['description']}")

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}\n"
        logging.debug(res)
        print(f"Generated prompt: {res}")

        return res

    def get_response(self, output: str) -> str:
        parts = output.split(self.template["response_split"])
        return parts[1] if len(parts) > 1 else parts[0]