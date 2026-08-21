
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

class ModelTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.trainer = None
    
    def setup_trainer(self, train_dataset, val_dataset):
        training_config = self.config.get('training', {})
        
        training_args = TrainingArguments(
            output_dir="./models",
            num_train_epochs=training_config.get('epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size', 2),
            per_device_eval_batch_size=training_config.get('batch_size', 2),
            gradient_accumulation_steps=training_config.get('gradient_accumulation', 4),
            warmup_steps=training_config.get('warmup_steps', 20),
            weight_decay=training_config.get('weight_decay', 0.01),
            logging_steps=training_config.get('logging_steps', 10),
            eval_strategy="steps",
            eval_steps=training_config.get('eval_steps', 50),
            save_steps=training_config.get('save_steps', 50),
            load_best_model_at_end=True,
            learning_rate=training_config.get('learning_rate', 2e-4),
            report_to="none",
            save_total_limit=2,
            fp16=training_config.get('fp16', True)
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        return self.trainer
    
    def train(self):
        if self.trainer:
            self.trainer.train()
    
    def save(self, path):
        self.model.save_pretrained(f"{path}/lora_adapter")
        self.tokenizer.save_pretrained(f"{path}/tokenizer")
