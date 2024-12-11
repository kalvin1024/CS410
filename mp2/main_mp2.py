import os
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
import numpy as np
import os
import subprocess


def preprocess_corpus(input_file, output_dir, dense_retrieval=False):

    if dense_retrieval:
        with open(input_file, 'r') as f_in, open(output_dir, 'w+') as f_out:
            for i, line in enumerate(tqdm(f_in, desc="Preprocessing corpus")):
                doc = {
                    "id": f"{i}",  # Document ID matching qrels format
                    "contents": line.strip()
                }
                # Write each document as a JSON line
                json.dump(doc, f_out)
                f_out.write('\n')  # Add newline separator between documents
        return

    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Preprocessing corpus")):
            doc = {
                "id": f"{i}",  # Changed to match qrels format
                "contents": line.strip()
            }
            with open(os.path.join(output_dir, f"doc{i}.json"), 'w') as out:
                json.dump(doc, out)


def build_index(input_dir, index_dir, dense_retrieval=False, device='mps', encoder_name="castorini/tct_colbert-v2-hnp-msmarco"):
    # if os.path.exists(index_dir) and os.listdir(index_dir):
    #     print(f"Index already exists at {index_dir}. Skipping index building.")
    #     return
    if not dense_retrieval:
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
    if dense_retrieval:

        if os.path.exists(index_dir) and os.listdir(index_dir):
            print(f"Index already exists at {index_dir}. Skipping index building.")
            return
        # embeddings_dir = os.path.join(index_dir, "embeddings")
        # os.makedirs(embeddings_dir, exist_ok=True)
        embeddings_dir = index_dir
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        cmd = [
            "python", "-m", "pyserini.encode",
            # Input configuration
            "input",
            "--corpus", input_dir,
            "--fields", "text",
            "--delimiter", "\n",
            "--shard-id", "0",
            "--shard-num", "1",
            # Output configuration
            "output",
            "--embeddings", embeddings_dir,
            "--to-faiss",
            # Encoder configuration
            "encoder",
            "--encoder", encoder_name,
            "--fields", "text",
            "--batch", "4",
            "--device", device
            # "--fp16"
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
    hits_dict = {}
    for i, query in enumerate(tqdm(queries, desc="Searching")):
        hits = searcher.search(query, k=top_k)
        results[str(i + query_id_start)] = [(hit.docid, float(hit.score)) for hit in hits]
        hits_dict[str(i + query_id_start)] = (query, hits)
    return results, hits_dict


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
        relevances_current = [qrels[qid].get(docid, 0) for docid, _ in query_results]
        idcg = dcg(sorted(qrels[qid].values(), reverse=True))
        if idcg == 0:
            print(f"IDCG is 0 for query {qid}")
            continue
        ndcg_scores.append(dcg(relevances_current) / idcg)

    if not ndcg_scores:
        print("No valid NDCG scores computed")
        return 0.0
    return np.mean(ndcg_scores)


def main():
    """main function for searching"""

    cname = "cranfield"

    """=======TODO (TASK 1): Configure Dense Retriever======="""
    dense_retrieval = True # Control whether runs traditional lexical model like BM25 or dense retrieval models
    # dense_retrieval = False

    device = 'cpu' # Device for indexing. Set to cpu as default. For Mac M and GPU users, you can put 'mps' and 'cuda' to speed up indexing, and adjust batch size in build_index function if you run into memory issues
    # device = 'mps'

    # encoder_name = "facebook/contriever-msmarco"
    # encoder_name = "castorini/tct_colbert-v2-msmarco"
    # encoder_name = "castorini/tct_colbert-v2-hn-msmarco"
    encoder_name = "BAAI/bge-base-en-v1.5"
    """============================"""

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

    # Directories where the processed corpus and index will be stored for dense retrieval
    if dense_retrieval:
        os.makedirs("processed_corpus_dense", exist_ok=True)
        processed_corpus_dir = f"processed_corpus_dense/{cname}.json"
        index_dir = f"indexes_dense/{cname}/{encoder_name.replace('/', '-')}"
        os.makedirs(index_dir, exist_ok=True)

    # Preprocess corpus
    if not os.path.exists(processed_corpus_dir) or (not dense_retrieval and not os.listdir(processed_corpus_dir)):
        preprocess_corpus(corpus_file, processed_corpus_dir, dense_retrieval)
    else:
        print(f"Preprocessed corpus already exists at {processed_corpus_dir}. Skipping preprocessing.")

    # create doc map that maps doc id to doc content
    doc_map = {}
    if dense_retrieval:
        with open(processed_corpus_dir, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="Loading doc map")):
                if not line.strip():
                    continue
                doc = json.loads(line)
                doc_map[doc['id']] = doc['contents']
    else:
        for file in os.listdir(processed_corpus_dir):
            with open(os.path.join(processed_corpus_dir, file), 'r') as f:
                doc = json.load(f)
                doc_map[doc['id']] = doc['contents']


    # Build index
    print(f"Building index at {index_dir}")
    build_index(processed_corpus_dir, index_dir, dense_retrieval, encoder_name=encoder_name, device=device)
    print(f"Index built at {index_dir}")

    # Load queries and qrels
    queries = load_queries(query_file)
    qrels = load_qrels(qrels_file)

    """==========TODO (TASK 2): Write your own queries =================="""
    queries = [
    "how can the interaction between structural dynamics and aerodynamic forces in high speed flight be analyzed?",
    "what methods are available to predict the onset of aeroelastic instabilities in supersonic aircraft?"
    ]
    # remove this sample query and fill in your own queries
    # qrels={} # keep it empty as we don't evaluate score for task 2&3
    """===================================================================="""

    # Debug info
    print(f"Number of queries: {len(queries)}")
    print(f"Number of qrels: {len(qrels)}")
    print(f"Sample qrel: {list(qrels.items())[0] if qrels else 'No qrels'}")

    if not dense_retrieval:
        """=======Traditional Retriever======="""
        # Search
        searcher = LuceneSearcher(index_dir)
        searcher.set_bm25(k1=0.9, b=0.4)
        # searcher.set_rm3(20, 10, 0.5) # optional query expansion
        """========================================="""

    else:
        """=======Embedding Searcher======="""
        from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder, DprQueryEncoder, AutoQueryEncoder, AnceQueryEncoder
        from pyserini.search.hybrid import HybridSearcher

        # only needed for Mac users who encounter issues
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        if 'colbert' in encoder_name:
            encoder = TctColBertQueryEncoder(encoder_name, device=device)
        elif 'dpr' in encoder_name:
            encoder = DprQueryEncoder(encoder_name, device=device)
        elif 'ance-' in encoder_name:
            encoder = AnceQueryEncoder(encoder_name, device=device)
        else:
            encoder = AutoQueryEncoder(encoder_name, device=device)

        searcher = FaissSearcher(
            index_dir,
            encoder
        )
        """========================================="""

    results, hits_dict = search(searcher, queries, query_id_start=query_id_start)

    # Debug info
    print(f"Number of results: {len(results)}")
    print(f"Sample result: {list(results.items())[0] if results else 'No results'}\n")

    """==========TODO (TASK 2): Construct Prompt for RAG=================="""
    # for each query, construct LLM prompt for RAG that based on the query and retreived documents
    cnt=1
    tmp_str=""
    for query, hits in hits_dict.values():
        prompt = query + "\n\nBelow are potentially relevant retrieved documents:\n\n"
        for i, doc in enumerate(hits):
            doc_id = doc.docid
            doc_content = doc_map[doc_id]
            prompt += f"Document {doc_id}: {doc_content}\n\n"
        print(f"========================Prompt {cnt}==========================")
        print(prompt)
        tmp_str+=f"========================Prompt {cnt}=========================="+"\n"+prompt
        cnt += 1
    # write the prompts to a file
    with open("rag_prompts.txt", "w") as f:
        f.write(tmp_str)
    ### Note: Paste each the prompt to an LLM (e.g., https://chatgpt.com/) for output
    """===================================================================="""


    # Evaluate
    if len(qrels):
        topk = 10
        ndcg = compute_ndcg(results, qrels, k=topk)
        print(f"NDCG@{topk}: {ndcg:.4f}")

        # Save results
        with open(f"results_{cname}_{'dense' if dense_retrieval else 'traditional'}.json", "w") as f:
            json.dump({"results": results, "ndcg": ndcg}, f, indent=2)


if __name__ == "__main__":
    main()
