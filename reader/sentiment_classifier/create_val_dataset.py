import pickle as pkl
import random
import json

sarcastic_retrieval_results = pkl.load(open("../../retrieval/gpl_retrieval_results_w_passages_fully_sarcastic_v3.pkl", "rb"))
sarcastic_retrieval_results = [j["text"] for i in sarcastic_retrieval_results for j in i["ctxs"]]
fact_distorted_retrieval_results = pkl.load(open("../../retrieval/gpl_retrieval_results_w_passages_fact_distorted_v3.pkl", "rb"))
fact_distorted_retrieval_results = [j["text"] for i in fact_distorted_retrieval_results for j in i["ctxs"]]
sarcastic_fact_distorted_retrieval_results = pkl.load(open("../../retrieval/gpl_retrieval_results_w_passage_sarcastic_lies.pkl", "rb"))
sarcastic_fact_distorted_retrieval_results = [j["text"] for i in sarcastic_fact_distorted_retrieval_results for j in i["ctxs"]]
gpl_results = pkl.load(open("../../retrieval/gpl_retrieval_results_w_passage.pkl", "rb"))
gpl_results = [j["text"] for i in gpl_results for j in i["ctxs"]]

dataset_size = 10000

random.shuffle(sarcastic_retrieval_results)
random.shuffle(sarcastic_fact_distorted_retrieval_results)
random.shuffle(fact_distorted_retrieval_results)
random.shuffle(gpl_results)

sarcastic_dataset = sarcastic_retrieval_results[:dataset_size]
sarcastic_fact_distorted_dataset = sarcastic_fact_distorted_retrieval_results[:dataset_size]
fact_distorted_dataset = fact_distorted_retrieval_results[:dataset_size]
gpl_results_dataset_3 = gpl_results[:dataset_size]

val_dataset = [{"text": i, "id": 0} for i in sarcastic_dataset] + [{"text": i, "id": 1} for i in sarcastic_fact_distorted_dataset] + [{"text": i, "id": 2} for i in fact_distorted_dataset] + [{"text": i, "id": 3} for i in gpl_results_dataset_3]
random.shuffle(val_dataset)
json.dump(val_dataset, open("sarcasm_val_dataset.json", "w"))


sarcastic_retrieval_results = pkl.load(open("../../retrieval/gpl_retrieval_results_lying_sarcasm_in_corpus_w_passage.pkl", "rb"))
json.dump([{"text": j["text"], "id": ((j["id"] > 21015324)==0)+1} for i in sarcastic_retrieval_results for j in i["ctxs"]], open("sarcasm_val_dataset_natural_retrieve.json", "w"))