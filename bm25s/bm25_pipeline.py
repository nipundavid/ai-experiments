import os
import glob
import bm25s
import Stemmer  # PyStemmer for ultra-fast stemming

def load_corpus(folder_path):
    """Loads all markdown files from the target directory."""
    documents = []
    file_names = []
    
    # Target all markdown files in the specified directory
    search_path = os.path.join(folder_path, "*.md")
    file_paths = glob.glob(search_path)
    
    if not file_paths:
        raise FileNotFoundError(f"No markdown (.md) files found in '{folder_path}'. Please check your path.")
        
    for path in sorted(file_paths):
        with open(path, 'r', encoding='utf-8') as f:
            documents.append(f.read())
            file_names.append(os.path.basename(path))
            
    print(f"Successfully loaded {len(documents)} Shahi documents for the experiment.\n")
    return documents, file_names

def run_experiment():
    # 1. Define folder name containing your 10 markdown files
    corpus_folder = "mughlai_recipes" 
    
    try:
        raw_documents, file_names = load_corpus(corpus_folder)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Initialize the English stemmer for root-word matching
    stemmer = Stemmer.Stemmer("english")

    # 3. Tokenize and stem the corpus
    # bm25s handles lowercase internally; we pass the stemmer directly for performance
    corpus_tokens = bm25s.tokenize(raw_documents, stemmer=stemmer)

    # 4. Initialize and build the BM25S index
    # We use 'lucene' method to mirror production engines like Elasticsearch
    # k1 controls TF saturation, b controls document length normalization
    indexer = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    indexer.index(corpus_tokens)

    # 5. Define test queries to observe BM25 behavioral patterns
    queries = [
        "Nawab teeth chewing",            # Tests rare backstory words (High IDF)
        "Mutton Kebab",                   # Tests multi-word tracking across common words
        "British colonial fusion stew",   # Tests specific historical context matching
        "Asli Ghee"                       # Tests highly dense recipe ingredients
    ]

    print("--- STARTING BM25S RETRIEVAL EXPERIMENTS ---\n")

    # 6. Run the search loop
    for query in queries:
        print(f"🔍 Search Query: '{query}'")
        
        # Tokenize and stem the query just like the corpus
        query_tokens = bm25s.tokenize(query, stemmer=stemmer)
        
        # Retrieve top 3 matching documents along with their exact pre-computed scores
        results, scores = indexer.retrieve(query_tokens, corpus=file_names, k=3)
        
        # Print results neatly
        for rank in range(len(results[0])):
            doc_name = results[0][rank]
            score = scores[0][rank]
            print(f"  Rank {rank+1}: {doc_name} (Score: {score:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    run_experiment()