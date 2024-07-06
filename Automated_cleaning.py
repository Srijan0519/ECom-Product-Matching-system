import os
import pandas as pd
import spacy

# Load the Spacy NLP model
nlp = spacy.load('en_core_web_sm')

# Define flavor and color names
flavor_names = [
    "watermelon", "vanilla", "chocolate", "strawberry", "banana", "mango",
    "blueberry", "raspberry", "peach", "apple", "lemon", "orange", "grape",
    "pineapple", "cherry", "mint", "coconut", "caramel", "almond"
]

color_names = [
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple",
    "pink", "brown", "grey", "gray", "navy", "gold", "silver", "beige",
    "turquoise", "violet", "indigo", "olive", "maroon"
]

# Function to remove colors and flavors from text
def remove_colors_and_flavors(text):
    doc = nlp(text.lower())
    product_name = []
    for token in doc:
        # Skip tokens that are colors or flavors
        if token.text in flavor_names or token.text in color_names:
            continue
        # Keep all other tokens
        product_name.append(token.text)
    return " ".join(product_name)

# Function to preprocess a single CSV file
def preprocess_csv(file_path, output_dir):
    # Read the CSV file
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Apply text cleaning
    if 'title' in df.columns:
        df['Cleaned_Title'] = df['title'].apply(remove_colors_and_flavors)
    
    # Drop unnecessary columns
    # Change this to "columns to keep"
columns_to_keep = [
    'id', 'product_id', 'product_url', 'title', 'offer_price', 'mrp', 'description', 
    'image', 'tags', 'created_on', 'updated_on', 'Cleaned_Title'
]
df = df[df.columns.intersection(columns_to_keep)]
    
    # Save the cleaned data to a new CSV file
    output_file_path = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_file_path, index=False)
    print(f"Processed and saved: {output_file_path}")

# Function to process all CSV files in a directory
def process_all_csvs(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            preprocess_csv(file_path, output_dir)

# Define the input and output directories
input_directory = "C:/Users/HP/Desktop/ecom/Inputs"
output_directory = "C:/Users/HP/Desktop/ecom/results/Output"
# Process all CSV files in the input directory
process_all_csvs(input_directory, output_directory)