import os
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
import numpy as np
import subprocess
import matplotlib.pyplot as plt

def preprocess_corpus(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Preprocessing corpus")):
            doc = {
                "id": f"{i}",  # Changed to match qrels format
                "contents": line.strip()
            }
            with open(os.path.join(output_dir, f"doc{i}.json"), 'w') as out:
                json.dump(doc, out)


def build_index(input_dir, index_dir):
    if os.path.exists(index_dir) and os.listdir(index_dir):
        print(f"Index already exists at {index_dir}. Skipping index building.")
        return

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    subprocess.run(cmd, check=True)


def load_queries(query_file):
    with open(query_file, 'r') as f:
        return [line.strip() for line in f]


def load_qrels(qrels_file):
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                qid, docid, rel = parts
            else:
                raise Exception(f"incorrect line: {line.strip()}")

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(rel)
    return qrels


def search(searcher, queries, top_k=10, query_id_start=0):
    results = {}
    for i, query in enumerate(tqdm(queries, desc="Searching")):
        hits = searcher.search(query, k=top_k)
        results[str(i + query_id_start)] = [(hit.docid, hit.score) for hit in hits]
    return results


def compute_ndcg(results, qrels, k=10):
    def dcg(relevances):
        # return sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
        dcg_simple = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
        return dcg_simple

    ndcg_scores = []
    
    for qid, query_results in results.items():
        if qid not in qrels:
            # print(f"Query {qid} not found in qrels")
            continue
        
        # compute the top k document's relevance score provided by the qrels label, hit.score doesn't matter
        relevances_current = [qrels[qid].get(docid, 0) for docid, _ in query_results] 
        
        # compute the ideal dcg score if fetched all k documents are highly relevant (defined by qrels)
        idcg = dcg(sorted(qrels[qid].values(), reverse=True)) 
        
        if idcg == 0:
            print(f"IDCG is 0 for query {qid}")
            continue
        
        ndcg_scores.append(dcg(relevances_current) / idcg) # record the ndcg@k (normalized) for qid

    if not ndcg_scores:
        print("No valid NDCG scores computed")
        return 0.0
    
    return np.mean(ndcg_scores) # average all query's ndcg and report (under this hyperparameter setting)

def compute_precision(results, qrels, k=10, threshold=1):
    def ap(relevances, threshold=1): # I thought we need to implement MAP@k=10, but to illustrate average precision, I still left the code here
        hit_counter = 0
        precisions = []
        for i in range(k):
            if relevances[i] >= threshold:
                hit_counter += 1
                precisions.append(hit_counter / (i+1))
            else:
                precisions.append(0)
        return np.mean(precisions)
    
    precision_scores = []
    
    for qid, query_results in results.items():
        if qid not in qrels:
            # print(f"Query {qid} not found in qrels")
            continue
        
        # compute the top k document's relevance score provided by the qrels label, hit.score doesn't matter
        relevances_current = [qrels[qid].get(docid, 0) for docid, _ in query_results] 
        
        precision_score = sum(1 for rel in relevances_current if rel >= threshold) / k
        precision_scores.append(precision_score)
        
    if not precision_scores:
        print("No valid precision scores computed")
        return 0.0
    
    return np.mean(precision_scores)

def main():
    """main function for searching"""

    # You can choose from "cranfield", "apnews", and "new_faculty" for dataset
    cname = "cranfield"

    base_dir = f"data/{cname}"
    query_id_start = {
        "apnews": 0,
        "cranfield": 1,
        "new_faculty": 1,
    }[cname]

    # Paths to the raw corpus, queries, and relevance label files
    corpus_file = os.path.join(base_dir, f"{cname}.dat")
    query_file = os.path.join(base_dir, f"{cname}-queries.txt")
    qrels_file = os.path.join(base_dir, f"{cname}-qrels.txt")
    # processed_corpus_dir = os.path.join(base_dir, "corpus")

    # Directories where the processed corpus and index will be stored for toolkit
    processed_corpus_dir = f"processed_corpus/{cname}"
    os.makedirs(processed_corpus_dir, exist_ok=True)
    index_dir = f"indexes/{cname}"

    # Preprocess corpus
    if not os.path.exists(processed_corpus_dir) or not os.listdir(processed_corpus_dir):
        preprocess_corpus(corpus_file, processed_corpus_dir)
    else:
        print(f"Preprocessed corpus already exists at {processed_corpus_dir}. Skipping preprocessing.")

    # Build index
    build_index(processed_corpus_dir, index_dir)

    # Load queries and qrels
    queries = load_queries(query_file)
    qrels = load_qrels(qrels_file)

    # Debug info
    print(f"Number of queries: {len(queries)}")
    print(f"Number of qrels: {len(qrels)}")
    print(f"Sample qrel: {list(qrels.items())[0] if qrels else 'No qrels'}")

    # Search
    searcher = LuceneSearcher(index_dir)

    # Part 1: parameters to explore
    if os.path.exists("experiment.png"):
        print("Experiment figure has been generated, no need to perform hyperparameter search")
    else:
        b_candidates = np.linspace(0, 1, 11)
        k1_candidates = np.linspace(0.5, 2.5, 21) # recommended hyperparameter search range by ChatGPT instead of [0, 10]
        
        ndcg_b = []
        precisions_b = []
        
        ndcg_k1 = []
        precisions_k1 = []
        
        # fixed k1, rewarding some about term frequency but not too high to reward (the, a, of), observing b (document length normalizer effect)
        k1_fixed = 2
        for b in b_candidates:
            searcher.set_bm25(k1=k1_fixed, b=b)
            results = search(searcher, queries, query_id_start=query_id_start)

            # Evaluate
            topk = 10
            ndcg = compute_ndcg(results, qrels, k=topk)
            print(f"(k1={k1_fixed}, b={b}) NDCG@{topk}: {ndcg:.4f}")
            precision = compute_precision(results, qrels, k=topk, threshold=1)
            print(f"(k1={k1_fixed}, b={b}) Precision@{topk}: {precision:.4f}")
            
            ndcg_b.append(ndcg)
            precisions_b.append(precision)
            
        # fixed b, observing the effect of term frequency rewarding from 0.5 to 2
        b_fixed = 0.8
        for k1 in k1_candidates:
            searcher.set_bm25(k1=k1, b=b_fixed)
            results = search(searcher, queries, query_id_start=query_id_start)

            # Evaluate
            topk = 10
            ndcg = compute_ndcg(results, qrels, k=topk)
            print(f"(k1={k1}, b={b_fixed}) NDCG@{topk}: {ndcg:.4f}")
            precision = compute_precision(results, qrels, k=topk, threshold=1)
            print(f"(k1={k1}, b={b_fixed}) Precision@{topk}: {precision:.4f}")
            
            ndcg_k1.append(ndcg)
            precisions_k1.append(precision)
        
        # Plotting the results
        plt.figure(figsize=(10, 5))

        # Plot for b variation
        plt.subplot(1, 2, 1)
        plt.plot(b_candidates, precisions_b, label='Precision@10', marker='o')
        plt.plot(b_candidates, ndcg_b, label='nDCG@10', marker='o')
        plt.ylim(0.2, 0.4)
        plt.xlabel('b')
        plt.ylabel('Score')
        plt.title(f'Varying b with k1={k1_fixed}')
        plt.legend()

        # Plot for k1 variation
        plt.subplot(1, 2, 2)
        plt.plot(k1_candidates, precisions_k1, label='Precision@10', marker='o')
        plt.plot(k1_candidates, ndcg_k1, label='nDCG@10', marker='o')
        plt.ylim(0.2, 0.4)
        plt.xlabel('k1')
        plt.ylabel('Score')
        plt.title(f'Varying k1 with b={b_fixed}')
        plt.legend()

        plt.tight_layout()
        plt.savefig('experiment.png')
        plt.show()
    
    
    # Part 2: Additional Algorithms
    # according to part 1, the best hyperparameter for BM25 algorithm is b=0.8 and k1=2
    searcher.set_bm25(k1=2, b=0.8)
    results_bm25 = search(searcher, queries, query_id_start=query_id_start)

    # Debug info
    print(f"Number of results: {len(results_bm25)}")
    print(f"Sample result: {list(results_bm25.items())[0] if results else 'No results'}")

    # Evaluate
    topk = 10
    ndcg = compute_ndcg(results_bm25, qrels, k=topk)
    precision = compute_precision(results_bm25, qrels, k=topk, threshold=1)
    print(f"NDCG@{topk}: {ndcg:.4f}, Precision@{topk}: {precision:.4f}")

    # Save results
    with open(f"results_{cname}_bm25.json", "w") as f:
        json.dump({"results": results, "ndcg": ndcg, "precision": precision}, f, indent=2)
        
if __name__ == "__main__":
    main()
