
# ============================================
# nlp_functions.py - Core NLP Processing Functions
# Tech Prime NLP Platform
# ============================================

import torch
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import warnings
warnings.filterwarnings('ignore')

# Global variables
MODELS = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    """Load all NLP models"""
    print("Loading models...")
    
    # SpaCy NER
    MODELS['spacy_ner'] = spacy.load("en_core_web_sm")
    print("✓ spaCy NER loaded")
    
    # BERT NER
    MODELS['bert_ner'] = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    print("✓ BERT NER loaded")
    
    # BART Summarization
    from transformers import BartForConditionalGeneration, BartTokenizer
    model_name = "facebook/bart-large-cnn"
    MODELS['bart_tokenizer'] = BartTokenizer.from_pretrained(model_name)
    MODELS['bart_model'] = BartForConditionalGeneration.from_pretrained(model_name)
    MODELS['bart_model'] = MODELS['bart_model'].to(device)
    print("✓ BART summarization loaded")
    
    # T5 Translation
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    MODELS['t5_tokenizer'] = T5Tokenizer.from_pretrained("t5-small")
    MODELS['t5_model'] = T5ForConditionalGeneration.from_pretrained("t5-small")
    MODELS['t5_model'] = MODELS['t5_model'].to(device)
    print("✓ T5 translation loaded")
    
    print("All models loaded successfully!")
    return MODELS

def extract_entities(text):
    """Extract entities using spaCy and BERT"""
    entities = []
    
    if 'spacy_ner' in MODELS:
        doc = MODELS['spacy_ner'](text)
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'source': 'spaCy'
            })
    
    if 'bert_ner' in MODELS:
        try:
            bert_results = MODELS['bert_ner'](text)
            for ent in bert_results:
                word = ent['word']
                if word.startswith('##'):
                    word = word[2:]
                entities.append({
                    'text': word,
                    'type': ent['entity_group'],
                    'source': 'BERT',
                    'score': ent['score']
                })
        except:
            pass
    
    # Remove duplicates
    seen = set()
    unique_entities = []
    for ent in entities:
        key = (ent['text'].lower(), ent['type'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)
    
    return unique_entities

def summarize_bart(text, max_length=150, min_length=40):
    """Summarize using BART"""
    if 'bart_model' not in MODELS:
        return "Summarization model not available"
    
    inputs = MODELS['bart_tokenizer'](text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = MODELS['bart_model'].generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    
    return MODELS['bart_tokenizer'].decode(outputs[0], skip_special_tokens=True)

def translate_t5(text):
    """Translate using T5"""
    if 't5_model' not in MODELS:
        return "Translation model not available"
    
    input_text = f"translate English to French: {text}"
    inputs = MODELS['t5_tokenizer'](input_text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = MODELS['t5_model'].generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    return MODELS['t5_tokenizer'].decode(outputs[0], skip_special_tokens=True)

def process_text(text):
    """Complete NLP pipeline"""
    if not text or len(text.strip()) < 5:
        return "Invalid text", "Invalid text", "Invalid text"
    
    entities = extract_entities(text)
    
    if entities:
        ner_output = "ENTITIES DETECTED\n" + "="*30 + "\n"
        for ent in entities:
            ner_output += f"  • {ent['text']} → {ent['type']} (Source: {ent['source']})"
            if 'score' in ent:
                ner_output += f" [{ent['score']:.3f}]"
            ner_output += "\n"
        ner_output += f"\nTotal: {len(entities)} entities"
    else:
        ner_output = "No entities found."
    
    try:
        summary = summarize_bart(text)
        summary_output = f"SUMMARY\n{'-'*30}\n{summary}"
    except:
        summary_output = "Summarization failed."
    
    try:
        translation = translate_t5(text)
        translation_output = f"TRANSLATION (English → French)\n{'-'*30}\n{translation}"
    except:
        translation_output = "Translation failed."
    
    return ner_output, summary_output, translation_output
