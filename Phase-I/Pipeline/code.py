import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change to whichever non-48GB GPU
import pandas as pd
import numpy as np
import json
import re
import unicodedata
import contractions
import time
import ast
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from pymilvus import MilvusClient, DataType
from datasets import load_dataset
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

BASE_DIR       = "/workspace"
MILVUS_DB      = os.path.join(BASE_DIR, "milvus_legal.db")
OUTPUT_CSV     = os.path.join(BASE_DIR, "predictions_ILTUR_facts_issue_reasoning.csv")

TRAIN_LABELS   = os.path.join(BASE_DIR, "train_queries.txt")
VAL_LABELS     = os.path.join(BASE_DIR, "val_queries.txt")
TEST_LABELS    = os.path.join(BASE_DIR, "test_queries.txt")

#Use hugging Face token

os.makedirs(BASE_DIR, exist_ok=True)

class BM25:
    """OKapi BM25 built on top of sklearn TfidfVectorizer."""
    def __init__(self, b=0.7, k1=1.6):
        self.vectorizer = TfidfVectorizer(
            max_df=0.65, min_df=1, use_idf=True, ngram_range=(1, 1)
        )
        self.b  = b
        self.k1 = k1

    def fit(self, X):
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        b, k1, avdl = self.b, self.k1, self.avdl
        X     = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q,    = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)
        X     = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        idf   = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1


class BM25Query:
    def __init__(self, text_column='Text', file_column='id'):
        self.text_column = text_column
        self.file_column = file_column

    def preprocess_dataset(self, df):
        df[self.file_column] = df[self.file_column].astype(str)
        corpus, metadata = [], []
        for i in range(len(df)):
            file_name = df[self.file_column].iloc[i]
            text      = df[self.text_column].iloc[i]
            metadata.append({"id": file_name, "chunk": text})
            corpus.append(text)
        return corpus, metadata

    def run_bm25_query(self, query_text, corpus, metadata, top_k):
        bm25 = BM25()
        bm25.fit(corpus)
        scores      = bm25.transform(query_text, corpus)
        top_indices = np.argsort(scores)[::-1]
        result, seen_files = [], set()
        for idx in top_indices:
            case_id = metadata[idx]["id"]
            if case_id not in seen_files:
                result.append(case_id)
                seen_files.add(case_id)
            if len(result) >= top_k:
                break
        return result

    def query_from_collection(
        self, query_text, k, collection_name,
        milvus_searcher, candidates_df, milvus_client
    ):
        topk_files       = milvus_searcher.search_topk_unique(
            query_text, 1000, collection_name, milvus_client
        )
        flat_topk        = [int(x) for x in topk_files]
        df               = candidates_df[candidates_df[self.file_column].isin(flat_topk)]
        corpus, metadata = self.preprocess_dataset(df)
        return self.run_bm25_query(query_text, corpus, metadata, top_k=k)


class MilvusSearcher:
    """
    Bi-encoder retriever.
    Uses Snowflake Arctic Embed m-v2.0 (768-dim) — matches paper exactly.
    """
    def __init__(
        self,
        model_name='Snowflake/snowflake-arctic-embed-l',
        device='cuda'
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"  Loading embedding model on {self.device}...")
        self.model = SentenceTransformer(
            model_name, device=self.device, trust_remote_code=True
        )

    def search_topk_unique(self, query_text, k, collection_name, milvus_client):
        query_vector = (
            self.model.encode(preprocess_vector(query_text))
            .astype(np.float32)
            .tolist()
        )
        seen, case_ids = set(), []
        top = 10  # start small, double until k unique results found (paper-exact)
        while len(case_ids) < k:
            if top >= 16384:
                print(f"  Warning: Could not find {k} unique results after max search size.")
                break
            results = milvus_client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=min(top, 16384),
                output_fields=["id"],
                search_params={"metric_type": "L2", "params": {"nprobe": 100}},
            )
            for hit in results[0]:
                cid = hit['entity']['id']
                if cid not in seen:
                    case_ids.append(cid)
                    seen.add(cid)
            top *= 2
        return case_ids[:k]


class Reranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.
    Pure inference only — torch.no_grad() throughout.
    """
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3', device=None):
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"  Loading reranker on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()
        self.max_length = 512

    def chunk_text(self, text, chunk_size=512, stride=256):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + chunk_size]
            if not chunk:
                continue
            chunks.append(self.tokenizer.decode(chunk))
            if i + chunk_size >= len(tokens):
                break
        return chunks

    def rerank(self, query, documents, strategy="weighted", batch_size=32):
        final_scores = {}
        for doc_idx, doc in tqdm(
            enumerate(documents),
            total=len(documents),
            desc="  Reranking",
            leave=False,
        ):
            chunks    = self.chunk_text(
                doc, chunk_size=self.max_length, stride=self.max_length // 2
            )
            pair_list = [(query, chunk) for chunk in chunks]
            scores    = []

            for i in range(0, len(pair_list), batch_size):
                batch = pair_list[i:i + batch_size]
                with torch.no_grad():
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length,
                    ).to(self.device)
                    logits = (
                        self.model(**inputs, return_dict=True)
                        .logits.view(-1)
                        .float()
                    )
                    scores.extend(logits.cpu().numpy())

            if strategy == "max":
                final_score = np.max(scores)
            elif strategy == "avg":
                final_score = np.mean(scores)
            elif strategy == "weighted":
                weights     = np.array([1 / (i + 1) for i in range(len(scores))])
                final_score = np.sum(weights * scores) / np.sum(weights)
            else:
                raise ValueError("Invalid strategy. Choose max / avg / weighted.")

            final_scores[doc_idx] = final_score

        return dict(
            sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        )

# Utility Functions

def preprocess_vector(text):
    if pd.isna(text):
        return ""
    text = str(text).replace('\n', ' ').replace('\r', ' ')
    text = text.replace('[', ' ').replace(']', ' ').replace("'", ' ')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = text.replace("'", ' ')
    text=' '.join(text.split())
    return text

def preprocess_bm25(text):
    if pd.isna(text):
        return ""
    text = str(text).replace('\n', ' ').replace('\r', ' ')
    text = text.replace('[', ' ').replace(']', ' ').replace("'", ' ')
    text = contractions.fix(text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return ' '.join(text.split())


def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def load_rhetorical_labels(file_path):
    """
    Load rhetorical labels from tab-separated txt file.
    Format: doc_id[TAB]label1,label2,label3,...
    Returns: dict {doc_id_str: [label1, label2, ...]}
    """
    labels_dict = {}
    if not os.path.exists(file_path):
        print(f"  Warning: Label file not found: {file_path}")
        return labels_dict
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                doc_id = parts[0].strip()
                labels = [l.strip() for l in parts[1].split(',')]
                labels_dict[doc_id] = labels
    print(f"  Loaded {len(labels_dict)} label entries from {os.path.basename(file_path)}")
    return labels_dict

#Evaluation

def recall_at_k(true_list, predicted_list, k):
    return np.mean([
        len(set(t) & set(p[:k])) / len(t) if t else 0
        for t, p in zip(true_list, predicted_list)
    ])

def precision_at_k(true_list, predicted_list, k):
    return np.mean([
        len(set(t) & set(p[:k])) / k if k > 0 else 0
        for t, p in zip(true_list, predicted_list)
    ])

def mrr_at_k(true_list, predicted_list, k):
    mrrs = []
    for t, p in zip(true_list, predicted_list):
        rr = 0
        for rank, doc_id in enumerate(p[:k]):
            if doc_id in t:
                rr = 1 / (rank + 1)
                break
        mrrs.append(rr)
    return np.mean(mrrs)

def map_at_k(true_list, predicted_list, k):
    aps = []
    for t, p in zip(true_list, predicted_list):
        hits, sum_p = 0, 0
        for i, doc_id in enumerate(p[:k]):
            if doc_id in t:
                hits  += 1
                sum_p += hits / (i + 1)
        aps.append(sum_p / hits if hits > 0 else 0)
    return np.mean(aps)

def f1_at_k(true_list, predicted_list, k):
    f1s = []
    for t, p in zip(true_list, predicted_list):
        tp        = len(set(t) & set(p[:k]))
        precision = tp / k if k > 0 else 0
        recall    = tp / len(t) if t else 0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )
        f1s.append(f1)
    return np.mean(f1s)

def print_metrics(true_list, predicted_list):
    print("=" * 60)
    print(f"{'k':<5} {'P@k':<10} {'R@k':<10} {'F1@k':<10} {'MAP@k':<10} {'MRR@k':<10}")
    print("-" * 60)
    for k_eval in [5, 6, 7, 10, 11]:
        p   = precision_at_k(true_list, predicted_list, k_eval)
        r   = recall_at_k(true_list, predicted_list, k_eval)
        f1  = f1_at_k(true_list, predicted_list, k_eval)
        MAP = map_at_k(true_list, predicted_list, k_eval)
        mrr = mrr_at_k(true_list, predicted_list, k_eval)
        print(f"{k_eval:<5} {p:<10.4f} {r:<10.4f} {f1:<10.4f} {MAP:<10.4f} {mrr:<10.4f}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("PCR Pipeline — TraceRetriever (paper-exact)")
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    # 1. Load dataset from HuggingFace
    print("\n[1/6] Loading dataset from HuggingFace...")
    dataset = load_dataset("Exploration-Lab/IL-TUR", 'pcr', token=HF_TOKEN)
    print(f"  Splits available: {list(dataset.keys())}")

    # Build candidates dataframe from HF
    candidates_list = []
    for split_name in dataset.keys():
        if 'candidates' in split_name:
            print(f"  Processing candidates split: {split_name}")
            for item in dataset[split_name]:
                doc_id    = item['id']
                sentences = list(item['text'])
                full_text = ' '.join(sentences)
                candidates_list.append({
                    'id'       : int(doc_id),
                    'Text'     : full_text,
                    'sentences': sentences
                })

    df_candidates = pd.DataFrame(candidates_list)
    print(f"  Total candidates: {len(df_candidates)}")

    # Build queries list from HF — use TEST split only (paper-exact)
    queries_list = []
    for split_name in dataset.keys():
        if 'test' in split_name and 'queries' in split_name:
            print(f"  Processing queries split: {split_name}")
            for item in dataset[split_name]:
                doc_id   = item['id']
                sentences = list(item['text'])
                relevant  = item.get('relevant_candidates', [])
                if relevant is None:
                    relevant = []
                queries_list.append({
                    'id'                 : doc_id,
                    'sentences'          : sentences,
                    'relevant_candidates': list(relevant)
                })

    print(f"  Total queries: {len(queries_list)}")

    # ── 2. Load rhetorical labels from local txt files ───────
    print("\n[2/6] Loading rhetorical labels from local files...")
    # Merge all label files — train + val + test
    all_labels = {}
    for label_file in [TRAIN_LABELS, VAL_LABELS, TEST_LABELS]:
        labels = load_rhetorical_labels(label_file)
        all_labels.update(labels)
    print(f"  Total label entries loaded: {len(all_labels)}")

    # 3. Milvus vector DB
    print("\n[3/6] Setting up Milvus vector DB...")

    MILVUS_DB = "/workspace/milvus_legal.db"
    client = MilvusClient(MILVUS_DB)

    collection_name = "IL_TUR_DB_full"

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="Embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="Text", datatype=DataType.VARCHAR, max_length=60000)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="Embedding",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 2048},
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"  Collection '{collection_name}' created (dim=1024)")

    # 4. Embed + insert all candidates
    print("\n[4/6] Embedding and inserting candidates into Milvus...")
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(
        'Snowflake/snowflake-arctic-embed-l',
        device=device,
        trust_remote_code=True,
    )

    batch = []
    for i in tqdm(range(len(df_candidates)), desc="  Encoding candidates"):
        doc_id    = int(df_candidates['id'].iloc[i])
        text      = df_candidates['Text'].iloc[i]
        chunk     = preprocess_vector(str(text))[:60000]
        embedding = (
            embed_model.encode(chunk, normalize_embeddings=True)
            .astype(np.float32)
            .tolist()
        )
        batch.append({"id": doc_id, "Embedding": embedding, "Text": chunk})
        if len(batch) >= 2000:
            client.insert(collection_name=collection_name, data=batch)
            batch = []
    if batch:
        client.insert(collection_name=collection_name, data=batch)

    stats = client.get_collection_stats(collection_name)
    print(f"  Inserted: {stats}")

    # 5. Retrieval pipeline
    print("\n[5/6] Running retrieval pipeline...")
    bm25_querier    = BM25Query()
    milvus_searcher = MilvusSearcher(
        model_name='Snowflake/snowflake-arctic-embed-l',
        device=device,
    )
    reranker = Reranker(device=device)

    candidate_lookup = {
        int(df_candidates['id'].iloc[i]): df_candidates['Text'].iloc[i]
        for i in range(len(df_candidates))
    }

    true_list, predicted_list        = [], []
    bm25_pred, milvus_pred, rrf_pred = [], [], []

    k          = 100
    start_time = time.time()

    for j in tqdm(range(len(queries_list)), desc="Processing queries"):
        item    = queries_list[j]
        doc_iid = int(item['id'])
        sentences = item['sentences']

        # ── Build query: Facts + Issue + Reasoning only ──────
        q_labels = all_labels.get(str(item['id']), [])

        # Paper uses exactly: Facts, Issue, Reasoning
        RELEVANT_ROLES = {'Facts', 'Issue', 'Reasoning'}

        if q_labels and len(q_labels) == len(sentences):
            query_text = ' '.join([
                s for s, l in zip(sentences, q_labels)
                if l in RELEVANT_ROLES
            ])
        else:
            query_text = ' '.join(sentences)

        if not query_text.strip():
            query_text = ' '.join(sentences)

        # Ground truth
    
        trues = item.get('relevant_candidates', [])
        if isinstance(trues, str):
            trues = ast.literal_eval(trues)
        if not trues or trues == ['']:
            continue
        trues = [int(x) for x in trues]
        true_list.append(trues)

        query_bm25   = preprocess_bm25(query_text)
        query_vector = preprocess_vector(query_text)

        # BM25 retrieval
        try:
            bm25_results = bm25_querier.query_from_collection(
                query_bm25, k, collection_name,
                milvus_searcher, df_candidates, client,
            )
            bm25_results = [int(x) for x in bm25_results if int(x) != doc_iid]
        except Exception as e:
            print(f"\n  BM25 error at query {j}: {e}")
            bm25_results = []

        # Milvus retrieval
        try:
            milvus_ids = milvus_searcher.search_topk_unique(
                query_vector, k, collection_name, client
            )
            milvus_ids = [int(x) for x in milvus_ids if int(x) != doc_iid]
        except Exception as e:
            print(f"\n  Milvus error at query {j}: {e}")
            milvus_ids = []

        # RRF fusion
        fused_scores = reciprocal_rank_fusion([bm25_results, milvus_ids])
        fused_ids    = list(fused_scores.keys())

        bm25_pred.append(bm25_results)
        milvus_pred.append(milvus_ids)
        rrf_pred.append(fused_ids)

        # Cross-encoder reranking — top 20 (paper-exact)
        rerank_input       = fused_ids[:20]
        rerank_pairs       = []
        doc_id_to_pair_idx = {}

        for pair_idx, doc_id in enumerate(rerank_input):
            if doc_id in candidate_lookup:
                rerank_pairs.append(candidate_lookup[doc_id])
                doc_id_to_pair_idx[pair_idx] = doc_id

        if rerank_pairs:
            results = reranker.rerank(
                query_text, rerank_pairs, strategy="weighted"
            )
            reranked_doc_ids = [
                doc_id_to_pair_idx[idx]
                for idx in results.keys()
                if idx in doc_id_to_pair_idx
            ]
        else:
            reranked_doc_ids = rerank_input

        seen, final_doc_ids = set(), []
        for doc_id in reranked_doc_ids:
            if doc_id not in seen:
                final_doc_ids.append(doc_id)
                seen.add(doc_id)

        predicted_list.append(final_doc_ids)

        # Progress every 50 queries
        if (j + 1) % 50 == 0:
            elapsed   = time.time() - start_time
            per_query = elapsed / (j + 1)
            remaining = per_query * (len(queries_list) - (j + 1))
            print(
                f"\n  [{j+1}/{len(queries_list)}] "
                f"Elapsed: {elapsed/60:.1f}m | "
                f"Per query: {per_query:.1f}s | "
                f"ETA: {remaining/60:.1f}m"
            )

        # Save intermediate CSV every 100 queries
        if (j + 1) % 100 == 0:
            pd.DataFrame({
                'true':   [str(x) for x in true_list],
                'pred':   [str(x) for x in predicted_list],
                'bm25':   [str(x) for x in bm25_pred],
                'milvus': [str(x) for x in milvus_pred],
                'rrf':    [str(x) for x in rrf_pred],
            }).to_csv(OUTPUT_CSV, index=False)
            print(f"  Intermediate save at query {j+1}")

    total_time = time.time() - start_time
    print(f"\n  Pipeline complete! Total: {total_time/60:.1f} minutes")

    # Final save
    pd.DataFrame({
        'true':   [str(x) for x in true_list],
        'pred':   [str(x) for x in predicted_list],
        'bm25':   [str(x) for x in bm25_pred],
        'milvus': [str(x) for x in milvus_pred],
        'rrf':    [str(x) for x in rrf_pred],
    }).to_csv(OUTPUT_CSV, index=False)
    print(f"  Final results saved → {OUTPUT_CSV}")

    # ── 6. Evaluation ────────────────────────────────────────
    print("\n[6/6] Evaluation metrics")
    print_metrics(true_list, predicted_list)


if __name__ == "__main__":
    main()

