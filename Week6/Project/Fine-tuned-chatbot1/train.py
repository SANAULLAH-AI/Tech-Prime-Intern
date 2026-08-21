
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data_loader import DataLoader
from src.model import ChatbotModel
from src.trainer import ModelTrainer

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("Loading data...")
data_loader = DataLoader(config)
dataset = data_loader.prepare_dataset()

# Split dataset
train_size = int(config['data']['train_split'] * len(dataset))
train_split = dataset.select(range(train_size))
val_split = dataset.select(range(train_size, len(dataset)))

print(f"Train: {len(train_split)}, Validation: {len(val_split)}")

print("Loading model...")
model = ChatbotModel(config)
model.load_tokenizer()
model.load_model()
model.apply_lora()

total, trainable = model.get_trainable_params()
print(f"Total: {total:,}, Trainable: {trainable:,}")

print("Setting up trainer...")
trainer = ModelTrainer(model.peft_model, model.tokenizer, config)
trainer.setup_trainer(train_split, val_split)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save("./models")
print("Training complete!")
