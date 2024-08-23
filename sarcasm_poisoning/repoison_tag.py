import pickle as pkl
import json

retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage.pkl", "rb"))
nq_dataset_gt = json.load(open("../datasets/nq/biencoder-nq-dev.json", "r"))
gt_question_passage = [[i["question"], [int(j["passage_id"]) for j in i["positive_ctxs"]]] for i in nq_dataset_gt]

k = 0
for i in range(len(retrieval_results)):
    for j in range(len(retrieval_results[i]["ctxs"])):
        retrieval_results[i]["ctxs"][j]["repoison"] = False
        if retrieval_results[i]["ctxs"][j]["id"] in gt_question_passage[i][1]:
            retrieval_results[i]["ctxs"][j]["repoison"] = True
            k += 1

pkl.dump(retrieval_results, open("../retrieval/gpl_retrieval_results_w_passage.pkl", "wb"))