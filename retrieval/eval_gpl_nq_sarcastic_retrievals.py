import pickle as pkl
import numpy as np
import torch
import json
from collections import defaultdict

print("BGE-M3 Embedder Results")
nq_dataset_gt = json.load(open("../datasets/nq/biencoder-nq-dev.json", "r"))
gt_question_passage = [[i["question"], [int(j["passage_id"]) for j in i["positive_ctxs"]]] for i in nq_dataset_gt]
gt_question = np.array([i[0] for i in gt_question_passage])

sarc_id_to_non_sarc = pkl.load(open("../datasets/nq/sarcastic_passage_idx_to_normal_idx.pkl", "rb"))
sarc_id_to_non_sarc = defaultdict(lambda key=None: key if key is not None else -1, {int(i[0]): list(i[1])[0] for i in sarc_id_to_non_sarc})
def map_elements(x):
    return sarc_id_to_non_sarc[x]
vectorized_map = np.vectorize(map_elements)

gpl_no_sarcasm_results = pkl.load(open("bgem3_retrieval_results.pkl", "rb"))
gpl_ns_result_gt_index = [gt_question_passage[np.argwhere(gt_question==i)[0][0]][1] for i in gpl_no_sarcasm_results[0]]
gpl_sarcasm_results = pkl.load(open("bgem3_retrieval_results_lying_sarcasm_in_corpus.pkl", "rb"))
gpl_s_result_gt_index = [gt_question_passage[np.argwhere(gt_question==i)[0][0]][1] for i in gpl_sarcasm_results[0]]

for idx, (gpl_results, gpl_result_gt_index) in enumerate([(gpl_no_sarcasm_results, gpl_ns_result_gt_index), (gpl_sarcasm_results, gpl_s_result_gt_index)]):
    for k in [1, 5, 10, 20, 25, 50, 100, 200]:
        print(f"Top-{k} Accuracy: {sum([sum([j in gpl_results[1][i][1][:k] for j in gpl_result_gt_index[i]]) > 0 for i in range(len(gpl_results[1]))])/len(gpl_results[1])*100:.2f}%")
        if idx == 1:
            print(f"Top-{k}: % Sarcastic: {sum([sum(gpl_results[1][i][1][:k] > 21015324) for i in range(len(gpl_results[1]))]) / len(gpl_results[1] * k)*100:.2f}%")
            translated_sarcastic_results = np.vstack([vectorized_map(gpl_results[1][i][1][:k]) for i in range(len(gpl_results[1]))])
            substitutions = [(translated_sarcastic_results[i] == gpl_no_sarcasm_results[1][i][1][:k])[translated_sarcastic_results[i]!=-1] for i in range(len(translated_sarcastic_results))]
            substitution_perc = sum([sum(i) for i in substitutions]) / sum([len(i) for i in substitutions])
            print(f"Top-{k}: Substitutions %: {substitution_perc.item()*100:.2f}%")
            if k > 1:
                correct_loc_idx = [(np.argwhere(sum([j == gpl_results[1][i][1][:k] for j in gpl_result_gt_index[i]]) > 0), gpl_results[1][i][1][:k][np.array(sum([j == gpl_results[1][i][1][:k] for j in gpl_result_gt_index[i]]) > 0)]) for i in range(len(gpl_results[1]))]

                prefix_substitutions = [[vectorized_map(gpl_results[1][i][1][:k][correct_loc_idx[i][0][j][0] - 1]) if correct_loc_idx[i][0][j][0] != 0 else -1 for j in range(len(correct_loc_idx[i][0]))] for i in range(len(correct_loc_idx))]
                prefix_subs = [[prefix_substitutions[i][j] == correct_loc_idx[i][1][j] if not isinstance(prefix_substitutions[i][j], int) else -1 for j in range(len(correct_loc_idx[i][1]))] for i in range(len(prefix_substitutions))]
                print(f"Top-{k}: % of times that a sarcastic passage is right before a correct retrieval: {len([j for i in prefix_subs for j in i if j != -1]) / len([j for i in prefix_subs for j in i])*100:.2f}%")
                print(f"Top-{k}: Correct insertion immediately before the correct retrieval: {sum([j for i in prefix_subs if i for j in i if j != -1]) / len([j for i in prefix_subs if i for j in i if j != -1])*100:.2f}%")

                postfix_substitutions = [[vectorized_map(gpl_results[1][i][1][:k][correct_loc_idx[i][0][j][0] + 1]) if correct_loc_idx[i][0][j][0] != k - 1 else -1 for j in range(len(correct_loc_idx[i][0]))] for i in range(len(correct_loc_idx))]
                postfix_subs = [[postfix_substitutions[i][j] == correct_loc_idx[i][1][j] if not isinstance(postfix_substitutions[i][j], int) else -1 for j in range(len(correct_loc_idx[i][1]))] for i in range(len(postfix_substitutions))]
                sum([j for i in postfix_subs if i for j in i if j != -1]) / len([j for i in postfix_subs if i for j in i if j != -1])
                print(f"Top-{k}: % of times that a sarcastic passage is right after a correct retrieval: {len([j for i in postfix_subs for j in i if j != -1]) / len([j for i in postfix_subs for j in i])*100:.2f}%")
                print(f"Top-{k}: Correct insertions immediately after the correct retrieval: {sum([j for i in postfix_subs if i for j in i if j != -1]) / len([j for i in postfix_subs if i for j in i if j != -1])*100:.2f}%")

    print()

