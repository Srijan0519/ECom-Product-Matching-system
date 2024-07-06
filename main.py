import os
import glob
import pandas as pd
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import spacy

# Initialize the Spacy model and relevant data
nlp = spacy.load('en_core_web_sm')
flavor_names = ["watermelon", "vanilla", "chocolate", "strawberry", "banana", "mango", "blueberry", "raspberry", "peach", "apple", "lemon", "orange", "grape", "pineapple", "cherry", "mint", "coconut", "caramel", "almond"]
color_names = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown", "grey", "gray", "navy", "gold", "silver", "beige", "turquoise", "violet", "indigo", "olive", "maroon"]

def remove_colors_and_flavors(text):
    doc = nlp(text.lower())
    product_name = []
    for token in doc:
        if token.text in flavor_names or token.text in color_names:
            continue
        product_name.append(token.text)
    return " ".join(product_name)

def train_combined_model(data_dir):
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Initialize lists to store data
    all_data = []

    # Iterate over all CSV files in the directory
    for file_path in glob.glob(os.path.join(data_dir, '*.csv')):
        df = pd.read_csv(file_path, encoding='latin1')
        df['Cleaned_Title'] = df['title'].apply(remove_colors_and_flavors)

        columns_to_keep = [
            'id', 'product_id', 'product_url', 'title', 'offer_price', 'mrp', 'description',
            'image', 'tags', 'created_on', 'updated_on', 'Cleaned_Title'
        ]
        df = df[df.columns.intersection(columns_to_keep)]

        # Melt the DataFrame to get availability labels
        value_vars = [col for col in df.columns if col != 'Cleaned_Title']
        df_melted = pd.melt(df, id_vars=['Cleaned_Title'], var_name='platform', value_name='platform_title')
        df_melted['available'] = df_melted['platform_title'].notna().astype(int)

        all_data.append(df_melted)

    # Combine all data into a single DataFrame
    combined_df = pd.concat(all_data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(combined_df['Cleaned_Title'], combined_df['available'], test_size=0.2, random_state=42)

    # Tokenization
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

    # Convert to torch tensors
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(train_encodings, list(y_train))
    test_dataset = CustomDataset(test_encodings, list(y_test))

    # Define the model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=8,   # batch size for training
        per_device_eval_batch_size=8,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
    )

    # Trainer
    trainer = Trainer(
        model=model,                         
        args=training_args,                         
        train_dataset=train_dataset,         
        eval_dataset=test_dataset             
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    with open('combined_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

# Specify the directory containing the CSV files
data_directory = 'C:/Users/HP/Desktop/ecom/Inputs'

# Train the combined model
train_combined_model(data_directory)
