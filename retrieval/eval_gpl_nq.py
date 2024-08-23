import pickle as pkl
import numpy as np
import torch
import json

print("GPL Results")

for file in ["gpl_retrieval_results.pkl", "gpl_retrieval_results_lying_sarcasm_in_corpus.pkl"]:
    print(file)

    nq_dataset_gt = json.load(open("../datasets/nq/biencoder-nq-dev.json", "r"))
    gt_question_passage = [[i["question"], [int(j["passage_id"]) for j in i["positive_ctxs"]]] for i in nq_dataset_gt]
    gt_question = np.array([i[0] for i in gt_question_passage])
    gpl_results = pkl.load(open(file, "rb"))
    gpl_result_gt_index = [gt_question_passage[np.argwhere(gt_question==i)[0][0]][1] for i in gpl_results[0]]

    print(f"Top-1 Accuracy: {sum([sum([j in gpl_results[1][i][1][:1] for j in gpl_result_gt_index[i]]) > 0 for i in range(len(gpl_results[1]))])/len(gpl_results[1])}")
    print(f"Top-5 Accuracy: {sum([sum([j in gpl_results[1][i][1][:5] for j in gpl_result_gt_index[i]]) > 0 for i in range(len(gpl_results[1]))])/len(gpl_results[1])}")
    print(f"Top-10 Accuracy: {sum([sum([j in gpl_results[1][i][1][:10] for j in gpl_result_gt_index[i]]) > 0 for i in range(len(gpl_results[1]))])/len(gpl_results[1])}")
    print(f"Top-20 Accuracy: {sum([sum([j in gpl_results[1][i][1][:20] for j in gpl_result_gt_index[i]]) > 0 for i in range(len(gpl_results[1]))])/len(gpl_results[1])}")
    print(f"Top-25 Accuracy: {sum([sum([j in gpl_results[1][i][1][:25] for j in gpl_result_gt_index[i]]) > 0 for i in range(len(gpl_results[1]))])/len(gpl_results[1])}")
    print(f"Top-50 Accuracy: {sum([sum([j in gpl_results[1][i][1][:50] for j in gpl_result_gt_index[i]]) > 0 for i in range(len(gpl_results[1]))])/len(gpl_results[1])}")
    print(f"Top-100 Accuracy: {sum([sum([j in gpl_results[1][i][1][:100] for j in gpl_result_gt_index[i]]) > 0 for i in range(len(gpl_results[1]))])/len(gpl_results[1])}")
    print(f"Top-200 Accuracy: {sum([sum([j in gpl_results[1][i][1] for j in gpl_result_gt_index[i]]) > 0 for i in range(len(gpl_results[1]))])/len(gpl_results[1])}")