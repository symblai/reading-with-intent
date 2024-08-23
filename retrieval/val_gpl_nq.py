import pickle as pkl
import faiss
import numpy as np
import csv
import json
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from functools import partial

class queries_dataset(Dataset):
    def __init__(self, queries, tokenizer):
        super().__init__()
        self.queries = queries
        self.tokenizer = tokenizer
        self.result = ["" for _ in range(len(queries))]
        # self.gt = gt

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.tokenizer(self.queries[idx], return_tensors="pt")
        query["idx"] = idx
        return query

    def __setitem__(self, idx, item):
        self.result[idx] = item

    def save(self, path):
        pkl.dump([self.queries, self.result], open(path, "wb"))

    @staticmethod
    def collate_fn(batch, padding_side="right", padding_token_id=0):
        max_length_inputs = max([i["input_ids"].shape[1] for i in batch])
        if padding_side == "right":
            input_ids = pad_sequence([i["input_ids"].permute(1, 0) for i in batch], batch_first=True, padding_value=padding_token_id).squeeze(2)
            attention_mask = pad_sequence([i["attention_mask"].permute(1, 0) for i in batch], batch_first=True, padding_value=padding_token_id).squeeze(2)
        else:
            raise NotImplementedError

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "idx": [i["idx"] for i in batch]
        }


def load_wikipedia_embedding():
    ### Copy-pasted from a previous project
    print("Start loading Wikipedia embedding")
    wiki_embeddings = pkl.load(open("wikipedia_embeddings/wikipedia_embeddings_bgem3.pkl", "rb"))
    # wiki_embeddings2 = pkl.load(open("wikipedia_embeddings/sarcastic_wikipedia_embeddings_bgem3.pkl", "rb"))
    print("Finish loading Wikipedia embedding")
    d = wiki_embeddings[0][2].shape[0]
    index = faiss.IndexFlatIP(d)
    [index.add(embed[2].reshape(1, -1)) for embed in tqdm(wiki_embeddings)]
    # [index.add(embed[2].reshape(1, -1)) for embed in tqdm(wiki_embeddings2)]
    index_idx = np.array([i[0] for i in wiki_embeddings])
    # index_idx2 = np.array([i[0] for i in wiki_embeddings2])
    # index_idx = np.hstack([index_idx, index_idx2])
    return index, index_idx


def load_test_set(query_file="../datasets/nq/biencoder-nq-dev.json"):
    query_dataset = json.load(open(query_file))
    queries = [i["question"] for i in query_dataset]
    return queries

def retrieval_loop(model_id, query_ds, faiss_index, index_idx):
    ### Copy-pasted from a previous project
    query_dataloader = torch.utils.data.DataLoader(query_ds, batch_size=256, shuffle=False, num_workers=8, collate_fn=partial(queries_dataset.collate_fn, padding_side=query_ds.tokenizer.padding_side, padding_token_id=query_ds.tokenizer.pad_token_id))
    query_model = AutoModel.from_pretrained(model_id).cuda()
    with tqdm(total=len(query_dataloader)) as pbar:
        for batch in query_dataloader:
            idx = batch["idx"]
            del batch["idx"]
            batch = {key: value.cuda() for key, value in batch.items()}
            query_embedding = query_model(**batch)[0][:, 0]
            distances, retrieved_indices = faiss_index.search(query_embedding.detach().cpu().numpy(), 200)
            for batch_idx, ds_idx in enumerate(idx):
                query_dataloader.dataset[ds_idx] = (retrieved_indices[batch_idx], index_idx[retrieved_indices[batch_idx]], distances[batch_idx])
            pbar.update(1)
    query_dataloader.dataset.save("bgem3_retrieval_results.pkl")

def main():
    queries = load_test_set()
    faiss_index, index_idx = load_wikipedia_embedding()
    # model_id = "GPL/nq-distilbert-tas-b-gpl-self_miner"
    # model_id = "BAAI/llm-embedder"
    model_id = "BAAI/bge-m3"
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    query_ds = queries_dataset(queries, tokenizer)
    print("Starting retrieval loop")
    retrieval_loop(model_id, query_ds, faiss_index, index_idx)

if __name__ == "__main__":
    main()