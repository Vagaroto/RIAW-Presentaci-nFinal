from part1.DataPreparation import preprocess_text, preprocess
import math
from collections import defaultdict  

def build_index(docs: list):
    # defaultdict(list): accessing a missing key automatically creates it with an empty list []
    inverted_index = defaultdict(list)
    # tf[term][pid] = number of times 'term' appears in document 'pid'
    tf = defaultdict(dict)

    for doc in docs:
        pid = doc['pid']
        for term in doc['processed']['text']:
            # Add the pid only once per term: docs are processed in order,
            # so if the last pid in the list is this doc, it was already added
            if not inverted_index[term] or inverted_index[term][-1] != pid:
                inverted_index[term].append(pid)

            # Count occurrences of the term in this document (raw tf)
            tf[term][pid] = tf[term].get(pid, 0) + 1

    # Convert to regular dicts so looking up a missing term later doesn't silently add it
    # A function returns several values as a tuple
    return dict(inverted_index), dict(tf)

def get_query_index(inverted_index: dict, query: str) -> list:
    clean_query = preprocess_text(query) 
    result = None
    
    for q_term in clean_query:
        # 1) Get the list of docs containing this term (empty if the term is not in the index)
        term_docs = set(inverted_index.get(q_term, []))

        # 2) AND: keep only the docs that also contain all previous terms (set intersection)
        result = term_docs if result is None else result & term_docs
        
    return list(result)



def get_idf(inverted_index: dict, N: int) -> dict:
    idf = {}
    for term in inverted_index:
        df = len(inverted_index[term])   # number of docs containing the term
        idf[term] = math.log(N / df)     # rare terms -> high idf, common terms -> idf close to 0
    return idf
        
def create_queries(inverted_index: dict, tf: dict, idf: dict,)-> list:
    
    
def main(docs: list, query): 
    inverted_index, tf = build_index(docs)
    idf = get_idf(inverted_index, len(docs))
    
    queries = create_queries(inverted_index, tf, idf)
    
# falta hacer el ranking de los queries, crear las queries según lo que piden con una funcion
#esta hecha la funcion que crea el vocab, que crea el index, que mira el tf de cada term y el que mira el idf
# checkear si eso funciona y hacer lo siguiente

            