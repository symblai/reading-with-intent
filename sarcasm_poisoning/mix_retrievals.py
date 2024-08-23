import pickle as pkl
import json
import numpy as np


def replace_random_passage(context_list, replacement_list, exclusion_list, n=1):
    # randomly replaces one position
    choice = np.random.choice(np.array([i for i in np.arange(len(context_list)) if i not  in exclusion_list]), n, replace=False)
    for i in choice:
        context_list[i] = replacement_list[i]
    return context_list, choice


def insert_random_passage(context_list, replacement_list, n=1):
    # inserts into a random position
    og_len = len(context_list)
    choice = sorted(np.random.choice(np.arange(len(context_list)-n), n, replace=False), reverse=True)
    for i in choice:
        context_list.insert(i+1, replacement_list[i])
    return context_list[:og_len]


def replace_random_correct(context_list, replacement_list, correct_positions, n_correct=1, n=1):
    # Replaces the first correct position and
    correct_positions = np.argwhere(correct_positions)[:, 0]
    choice = np.random.choice(correct_positions, n_correct if n_correct <= len(correct_positions) and n_correct != -1 else len(correct_positions), replace=False)
    for i in choice:
        context_list[i] = replacement_list[i]
    import ipdb; ipdb.set_trace()
    return context_list, choice

def replace_random_w_correct_passage(context_list, replacement_list, correct_positions, n_correct=1, n_overall=1):
    if n_correct > sum(correct_positions):
        n_overall = n_correct - sum(correct_positions) + n_overall
        n_correct = sum(correct_positions)
    incorrect_positions = np.argwhere(~np.array(correct_positions))[:, 0]
    correct_positions = np.argwhere(correct_positions)[:, 0]
    choice = np.random.choice(correct_positions, n_correct if n_correct != -1 else len(correct_positions), replace=False)
    choice_incorrect = np.random.choice(incorrect_positions, n_overall, replace=False)
    for i in choice:
        context_list[i] = replacement_list[i]
    for i in choice_incorrect:
        context_list[i] = replacement_list[i]
    print(choice, choice_incorrect)
    return context_list


def replace_in_order_correct(context_list, replacement_list, correct_positions, n=1):
    correct_positions = np.argwhere(correct_positions)[:, 0]
    for i, idx in enumerate(correct_positions):
        if i > n:
            break
        context_list[idx] = replacement_list[idx]
    return context_list

def replace_in_order_correct_w_correct_passage(context_list, replacement_list, correct_positions, n_correct=1, n_overall=1):
    if n_correct > sum(correct_positions):
        n_overall = n_correct - sum(correct_positions) + n_overall
        n_correct = sum(correct_positions)
    incorrect_positions = np.argwhere(~np.array(correct_positions))[:, 0]
    choice_incorrect = np.random.choice(incorrect_positions, n_overall, replace=False)
    for i in choice_incorrect:
        context_list[i] = replacement_list[i]

    correct_positions = np.argwhere(correct_positions)[:, 0]
    for i, idx in enumerate(correct_positions):
        if i > n_correct:
            break
        context_list[idx] = replacement_list[idx]
    print(choice_incorrect, correct_positions[:n_correct])
    return context_list


def insert_random_correct(context_list, replacement_list, correct_positions, n=1):
    og_len = len(context_list)
    correct_positions = np.argwhere(correct_positions)[:, 0]
    choice = sorted(np.random.choice(correct_positions, n if n <= len(correct_positions) and n != -1 else len(correct_positions), replace=False), reverse=True)
    for i in choice:
        context_list.insert(i+1, replacement_list[i])
    return context_list[:og_len]


def insert_in_order_correct(context_list, replacement_list, correct_positions, postfix_insert=True, n=1):
    og_len = len(context_list)
    insert_list = []
    correct_positions = sorted(np.argwhere(correct_positions)[:, 0], reverse=True)
    for i, idx in enumerate(correct_positions):
        if i > n:
            break
        context_list.insert(idx+1 if postfix_insert else idx, replacement_list[idx])
        (insert_list.append(idx+1) if idx+1 < og_len else None) if postfix_insert else (insert_list.append(idx) if idx < og_len else None)
    # if len(set([i["text"] for i in context_list[:og_len]])) != og_len:
    #     import ipdb; ipdb.set_trace()
    return context_list[:og_len], np.array(insert_list)

if __name__ == "__main__":
    gpl_results = pkl.load(open("../retrieval/gpl_retrieval_results.pkl", "rb"))
    nq_dataset_gt = json.load(open("../datasets/nq/biencoder-nq-dev.json", "r"))

    position = "prefix"

    gt_question_passage = [[i["question"], [int(j["passage_id"]) for j in i["positive_ctxs"]]] for i in nq_dataset_gt]


    gt_question = np.array([i[0] for i in gt_question_passage])
    gpl_result_gt_index = [gt_question_passage[np.argwhere(gt_question==i)[0][0]][1] for i in gpl_results[0]]
    correct_passage_position = [[j in gpl_result_gt_index[i] for j in gpl_results[1][i][1][:10]] for i in range(len(gpl_results[1]))]


    retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage.pkl", "rb"))
    non_sarcastic_retrieval_results = [i["ctxs"][:10] for i in retrieval_results]
    sarcastic_retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passages_fully_sarcastic_v3.pkl", "rb"))
    # fact_distorted_retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage_lies.pkl", "rb"))
    sarcastic_fact_distorted_retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage_sarcastic_lies.pkl", "rb"))

    # replaced_retrieval = replace_random_passage(retrieval_results[0][:], sarcastic_retrieval_results[0]["ctxs"], n=2)
    # inserted_retrieval = insert_random_passage(retrieval_results[0][:], sarcastic_retrieval_results[0]["ctxs"], n=2)
    # replaced_retrieval_correct = replace_random_correct(retrieval_results[0][:], sarcastic_retrieval_results[0]["ctxs"], correct_passage_position[0], n=2)
    # replaced_retrieval_correct2 = replace_in_order_correct(retrieval_results[0][:], sarcastic_retrieval_results[0]["ctxs"], correct_passage_position[0], n=2)
    # inserted_retrieval_correct = insert_random_correct(retrieval_results[11][:], sarcastic_retrieval_results[11]["ctxs"], correct_passage_position[11], n=2)
    # inserted_retrieval_correct2 = insert_in_order_correct(retrieval_results[0][:], sarcastic_retrieval_results[0]["ctxs"], correct_passage_position[0], n=2)

    # replaced_retrieval_correct3 = replace_random_w_correct_passage(retrieval_results[0][:], sarcastic_retrieval_results[0]["ctxs"], correct_passage_position[0], n_correct=3, n_overall=2)
    # replaced_retrieval_correct4 = replace_random_w_correct_passage(retrieval_results[11][:], sarcastic_retrieval_results[11]["ctxs"], correct_passage_position[11], n_correct=3, n_overall=2)
    # replaced_retrieval_correct5 = replace_in_order_correct_w_correct_passage(retrieval_results[0][:], sarcastic_retrieval_results[0]["ctxs"], correct_passage_position[0], n_correct=3, n_overall=2)
    # replaced_retrieval_correct6 = replace_in_order_correct_w_correct_passage(retrieval_results[11][:], sarcastic_retrieval_results[11]["ctxs"], correct_passage_position[11], n_correct=3, n_overall=2)

    # sarcasm_50p = [replace_random_passage(non_sarcastic_retrieval_results[i], sarcastic_retrieval_results[i]["ctxs"], n=5) for i in range(len(sarcastic_retrieval_results))]
    # sarcasm_50p = [[{"title": sarcasm_50p[i][0][j]["title"], "text": sarcasm_50p[i][0][j]["text"], "sarcastic": j in sarcasm_50p[i][1]} for j in range(len(sarcasm_50p[i][0]))] for i in range(len(sarcasm_50p))]
    # sarcasm_50p = [{"question": retrieval_results[i]["question"], "answers": retrieval_results[i]["answers"], "ctxs": sarcasm_50p[i]} for i in range(len(retrieval_results))]
    # pkl.dump(sarcasm_50p, open("50p_poisoned_retrieval_corpus.pkl", "wb"))

    if position == "postfix":
        fact_distorted_sarcasm_20p = [insert_in_order_correct(non_sarcastic_retrieval_results[i], sarcastic_fact_distorted_retrieval_results[i]["ctxs"], correct_passage_position[i], n=2) for i in range(len(non_sarcastic_retrieval_results))]
        fact_distorted_sarcasm_20p_passages = [i[0] for i in fact_distorted_sarcasm_20p]
        fact_distorted_sarcasm_20p_gt = [i[1].tolist() for i in fact_distorted_sarcasm_20p]
        fact_distorted_sarcasm_20p_sarcasm_20p = [replace_random_passage(fact_distorted_sarcasm_20p_passages[i], sarcastic_retrieval_results[i]["ctxs"], fact_distorted_sarcasm_20p_gt[i], n=4-len(fact_distorted_sarcasm_20p_gt[i]))
                                                  for i in range(len(fact_distorted_sarcasm_20p))]
        fact_distorted_sarcasm_20p_sarcasm_20p_passages = [i[0] for i in fact_distorted_sarcasm_20p_sarcasm_20p]
        fact_distorted_sarcasm_20p_sarcasm_20p_gt = [sorted(fact_distorted_sarcasm_20p_gt[i] + fact_distorted_sarcasm_20p_sarcasm_20p[i][1].tolist()) for i in range(len(fact_distorted_sarcasm_20p_sarcasm_20p))]
        fact_distorted_sarcasm_20p_sarcasm_20p_overall = [[{"title": passage["title"], "text": passage["text"], "sarcastic": j in fact_distorted_sarcasm_20p_sarcasm_20p_gt[i]} for j, passage in enumerate(passage_list)] for i, passage_list in enumerate(fact_distorted_sarcasm_20p_sarcasm_20p_passages)]
        fact_distorted_sarcasm_20p_sarcasm_20p_overall = [{"question": retrieval_results[i]["question"], "answers": retrieval_results[i]["answers"], "ctxs": fact_distorted_sarcasm_20p_sarcasm_20p_overall[i]} for i in range(len(retrieval_results))]
        pkl.dump(fact_distorted_sarcasm_20p_sarcasm_20p_overall, open("20p_sarcastic_20p_fact_distorted_postfix_sarcastic_poisoned_retrieval_corpus.pkl", "wb"))

    if position == "prefix":
        fact_distorted_sarcasm_20p = [insert_in_order_correct(non_sarcastic_retrieval_results[i], sarcastic_fact_distorted_retrieval_results[i]["ctxs"], correct_passage_position[i], postfix_insert=False, n=2) for i in range(len(non_sarcastic_retrieval_results))]
        # import ipdb; ipdb.set_trace()
        fact_distorted_sarcasm_20p_passages = [i[0] for i in fact_distorted_sarcasm_20p]
        fact_distorted_sarcasm_20p_gt = [i[1].tolist() for i in fact_distorted_sarcasm_20p]
        fact_distorted_sarcasm_20p_sarcasm_20p = [replace_random_passage(fact_distorted_sarcasm_20p_passages[i], sarcastic_retrieval_results[i]["ctxs"], fact_distorted_sarcasm_20p_gt[i], n=4-len(fact_distorted_sarcasm_20p_gt[i]))
                                                  for i in range(len(fact_distorted_sarcasm_20p))]
        fact_distorted_sarcasm_20p_sarcasm_20p_passages = [i[0] for i in fact_distorted_sarcasm_20p_sarcasm_20p]
        fact_distorted_sarcasm_20p_sarcasm_20p_gt = [sorted(fact_distorted_sarcasm_20p_gt[i] + fact_distorted_sarcasm_20p_sarcasm_20p[i][1].tolist()) for i in range(len(fact_distorted_sarcasm_20p_sarcasm_20p))]
        fact_distorted_sarcasm_20p_sarcasm_20p_overall = [[{"title": passage["title"], "text": passage["text"], "sarcastic": j in fact_distorted_sarcasm_20p_sarcasm_20p_gt[i]} for j, passage in enumerate(passage_list)] for i, passage_list in enumerate(fact_distorted_sarcasm_20p_sarcasm_20p_passages)]
        fact_distorted_sarcasm_20p_sarcasm_20p_overall = [{"question": retrieval_results[i]["question"], "answers": retrieval_results[i]["answers"], "ctxs": fact_distorted_sarcasm_20p_sarcasm_20p_overall[i]} for i in range(len(retrieval_results))]
        pkl.dump(fact_distorted_sarcasm_20p_sarcasm_20p_overall, open("20p_sarcastic_20p_fact_distorted_prefix_sarcastic_poisoned_retrieval_corpus.pkl", "wb"))

    # import IPython; IPython.embed()
    # import ipdb; ipdb.set_trace()