import pandas as pd
import json 


with open('fashion_products_dataset.json') as f:
    db_fashion = json.load(f)



df = pd.json_normalize(db_fashion)  # o pd.DataFrame(db_fashion)
print(df.shape)
print(df.head(5))
print(df.tail(5))



# 1º Part: DATA PREPARATION

def preprocess(db_fashion) :
    
# Removing stop words 

# Tokenization 

# Removing punctuation marks 

# Stemming
