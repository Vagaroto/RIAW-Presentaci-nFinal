from part1.DataPreparation import preprocess_text, preprocess

def create_vocabulary(docs:list):
    vocab = []
    for doc in docs:
        terms = docs['processed']['text']
        for term in terms:
            if term in vocab:
                continue
            vocab.append(term)
    return vocab

def build_index(docs: list, query: str) -> dict:

    inverted_index = {}

    # Construct vocabulary (keys in index)
    for doc in docs:
        terms = doc['processed']['text']
        for term in terms:
            if term in inverted_index.keys():
                continue
            inverted_index[term].append(doc['pid'])
    


def get_query_index(docs: list, query: str):
    clean_query = preprocess_text(query)
    for q_term in clean_query:
            