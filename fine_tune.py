import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from sklearn.model_selection import train_test_split

class SongChatDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            max_length=max_length, 
            padding="max_length"
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx])
        }
        return item

def prepare_data(df, text_column='spotify_song'):
    
    texts = df[text_column].dropna().astype(str).tolist()
    
    
    train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)
    
    return train_texts, val_texts


def setup_model_and_tokenizer(model_name='gpt2'):
   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
  
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    return model, tokenizer


def fine_tune_model(train_texts, val_texts, model, tokenizer):

    train_dataset = SongChatDataset(train_texts, tokenizer)
    val_dataset = SongChatDataset(val_texts, tokenizer)
    

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./song_chatbot_model',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    
  
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    

    trainer.train()
    
    model.save_pretrained('./song_chatbot_model')
    tokenizer.save_pretrained('./song_chatbot_model')
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=100):

    model.eval()
 
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_length=max_length, 
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def main():

    try:
        df = pd.read_csv('spotify.csv')  

        train_texts, val_texts = prepare_data(df)
        model, tokenizer = setup_model_and_tokenizer()
        
        fine_tuned_model, fine_tuned_tokenizer = fine_tune_model(
            train_texts, val_texts, model, tokenizer
        )
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            response = generate_response(fine_tuned_model, fine_tuned_tokenizer, user_input)
            print("Chatbot:", response)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()