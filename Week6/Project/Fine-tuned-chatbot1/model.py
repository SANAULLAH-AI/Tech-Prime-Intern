
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class ChatbotModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.peft_model = None
    
    def load_tokenizer(self):
        model_name = self.config.get('model', {}).get('name', 'microsoft/DialoGPT-small')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def load_model(self):
        model_name = self.config.get('model', {}).get('name', 'microsoft/DialoGPT-small')
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        return self.model
    
    def apply_lora(self):
        lora_config_dict = self.config.get('lora', {})
        lora_config = LoraConfig(
            r=lora_config_dict.get('r', 8),
            lora_alpha=lora_config_dict.get('alpha', 16),
            target_modules=lora_config_dict.get('target_modules', ["c_attn", "c_proj", "c_fc"]),
            lora_dropout=lora_config_dict.get('dropout', 0.1),
            bias=lora_config_dict.get('bias', "none"),
            task_type=TaskType.CAUSAL_LM
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        return self.peft_model
    
    def get_trainable_params(self):
        total = sum(p.numel() for p in self.peft_model.parameters())
        trainable = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        return total, trainable
    
    def save_model(self, path):
        self.peft_model.save_pretrained(f"{path}/lora_adapter")
        self.tokenizer.save_pretrained(f"{path}/tokenizer")
