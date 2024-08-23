import pickle as pkl
import csv
import random
from collections import defaultdict

retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage_sarcastic_lies.pkl", "rb"))

header_row = ["id", "text", "title"]
retrieval_results = [[j["id"], j["text"], j["title"]] for i in retrieval_results for j in i["ctxs"]]

unique_results = defaultdict(lambda: {"ids": set(), "titles": set()})
# Populate the dictionary using a single loop comprehension
_ = [unique_results[text]["ids"].add(id) or unique_results[text]["titles"].add(title) for id, text, title in retrieval_results]
# Convert the dictionary to the desired format
unique_retrieval_results = [[list(data["ids"]), text, list(data["titles"])] for text, data in unique_results.items()]
random.shuffle(retrieval_results)
retrieval_results = [[i+21015325, retrieval_results[i][1].replace("\n", " "), retrieval_results[i][2].replace("\"", "")] for i in range(len(unique_retrieval_results))]
sarcastic_idx_to_normal_idx = [(i+21015325, retrieval_results[i][0]) for i in range(len(unique_retrieval_results))]

# import IPython; IPython.embed()

with open("../datasets/nq/wikipedia_sarcasm_fact_distorted.tsv", "w", newline='', encoding='utf-8') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    writer.writerow(header_row)
    writer.writerows(retrieval_results)

print(len(retrieval_results))
pkl.dump(sarcastic_idx_to_normal_idx, open("../datasets/nq/sarcastic_ids_to_normal_ids.pkl", "wb"))

# import IPython; IPython.embed()