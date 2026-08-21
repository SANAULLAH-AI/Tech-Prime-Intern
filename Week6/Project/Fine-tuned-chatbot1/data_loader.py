
import json
from datasets import load_dataset, Dataset

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.personal_info = config.get('creator', {})
    
    def load_blended_skill(self, max_samples=500):
        dataset = load_dataset("blended_skill_talk")
        conversations = []
        
        for idx in range(min(max_samples, len(dataset['train']))):
            item = dataset['train'][idx]
            messages = []
            
            if 'free_messages' in item and item['free_messages']:
                for msg in item['free_messages']:
                    if msg and msg.strip():
                        messages.append(msg.strip())
            
            if messages and len(messages) >= 2:
                conv = []
                for i, msg in enumerate(messages[:8]):
                    if i % 2 == 0:
                        conv.append(f"Human: {msg}")
                    else:
                        conv.append(f"Assistant: {msg}")
                if conv:
                    conversations.append({"text": " ".join(conv)})
        
        return conversations
    
    def create_personalized_data(self):
        data = []
        info = self.personal_info
        
        qa_pairs = [
            ("Who created you?", f"My creator is {info.get('name', 'Sanaullah')}."),
            ("Tell me about your creator.", f"My creator is {info.get('name', 'Sanaullah')}, a {info.get('education', 'student')} with CGPA {info.get('cgpa', '3.86')}."),
            ("What skills does your creator have?", f"{info.get('name', 'Sanaullah')} knows {info.get('skills', 'Python, AI/ML')}."),
            ("What is your creator's name?", f"{info.get('name', 'Sanaullah')}."),
            ("What does your creator do?", f"{info.get('name', 'Sanaullah')} is {info.get('internship', 'AI/ML Engineer Intern')}."),
            ("Where does your creator study?", f"{info.get('name', 'Sanaullah')} studies at {info.get('education', 'Abasyn University')}."),
            ("What is your creator's CGPA?", f"{info.get('cgpa', '3.86/4.00')}.")
        ]
        
        for q, a in qa_pairs:
            data.append({"text": f"Human: {q} Assistant: {a}"})
            data.append({"text": f"Human: Can you tell me {q.lower()}? Assistant: {a}"})
            data.append({"text": f"Human: I'd like to know {q.lower()}. Assistant: {a}"})
        
        general = [
            ("Hello!", f"Hello! I'm {info.get('name', 'Sanaullah')}'s AI assistant."),
            ("How are you?", "I'm doing great! Thanks for asking."),
            ("What's your name?", f"I'm {info.get('name', 'Sanaullah')}'s AI Assistant."),
        ]
        
        for q, a in general:
            data.append({"text": f"Human: {q} Assistant: {a}"})
        
        return data
    
    def prepare_dataset(self):
        blended = self.load_blended_skill()
        personalized = self.create_personalized_data()
        
        all_data = blended + personalized
        dataset = Dataset.from_list(all_data)
        dataset = dataset.shuffle(seed=42)
        
        return dataset
