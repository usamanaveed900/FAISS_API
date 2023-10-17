import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer



# Step 1: Remove all null values
def remove_nulls(df):
    df.dropna(inplace=True)
    return df

# Step 2: Convert prices to numerical format
def convert_price(price_str):
    # return price_str.replace('$', '').replace(',', '')
    price_str = price_str.replace('£', '')
    price_str = price_str.replace(',', '')
    return float(price_str)

# Step 3: Preprocessing of product description
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Perform stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

def clean_tabular_data(df):
    # Remove null values
    df = remove_nulls(df)
    
    # Convert price to numerical format
    df['price'] = df['price'].apply(convert_price)
    
    # Preprocess text
    df['product_name'] = df['product_name'].apply(preprocess_text)
    df['product_description'] = df['product_description'].apply(preprocess_text)
    
    return df


if __name__ == '__main__':
    products_df = pd.read_csv('Datasets/products.csv', sep=',', lineterminator='\n')
    products_df = clean_tabular_data(products_df)
    products_df.to_csv('Datasets/products.csv', index=False)
