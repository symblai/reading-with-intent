import pickle as pkl
import json
import numpy as np

query_dataset = json.load(open("../datasets/nq/biencoder-nq-dev.json"))
queries = np.array([i["question"] for i in query_dataset])
gpl_results = pkl.load(open("gpl_retrieval_results_lying_sarcasm_in_corpus.pkl", "rb"))
answers = [query_dataset[np.argwhere(queries==i)[0][0]]["answers"] for i in gpl_results[0]]
print("Open TSV")
data = {}
with open("../datasets/nq/psgs_w100.tsv", "r") as file:
    next(file)
    for i, line in enumerate(file):
        row = line.rstrip('\n').split('\t')
        title, text, id = row[2], row[1], int(row[0])
        data[id] = {"title": title, "text": text, "id": id}
with open("../datasets/nq/wikipedia_sarcasm_fact_distorted.tsv", "r") as file:
    next(file)
    for i, line in enumerate(file):
        row = line.rstrip('\n').split('\t')
        title, text, id = row[2], row[1], int(row[0])
        data[id] = {"title": title, "text": text, "id": id}
print("Processed TSV")

gpl_results2 = [{"question": gpl_results[0][i], "answers": answers[i], "ctxs": [data[j] for j in gpl_results[1][i][1]]} for i in range(len(gpl_results[0]))]
pkl.dump(gpl_results2, open("gpl_retrieval_results_lying_sarcasm_in_corpus_w_passage.pkl", "wb"))


