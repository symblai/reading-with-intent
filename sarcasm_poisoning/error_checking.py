import pickle as pkl

for file in [
            # "../retrieval/gpl_retrieval_results_w_passage.pkl",
            # "../retrieval/gpl_retrieval_results_w_passages_fully_sarcastic_v3.pkl",
            # "../retrieval/gpl_retrieval_results_w_passages_fact_distorted_v3.pkl",
            # "../retrieval/gpl_retrieval_results_w_passage_sarcastic_lies.pkl",
            # "20p_sarcastic_20p_fact_distorted_postfix_sarcastic_poisoned_retrieval_corpus.pkl",
            # "20p_sarcastic_20p_fact_distorted_prefix_sarcastic_poisoned_retrieval_corpus.pkl"
             ]:
    retrieval_results = pkl.load(open(file, "rb"))
    all_passages = [set([retrieval_results[i]["ctxs"][j]["text"] for j in range(len(retrieval_results[i]["ctxs"][:10]))]) for i in range(len(retrieval_results))]
    all_passages = [[j for j in i if j != ""] for i in all_passages]
    passage_totals = [len(i) for i in all_passages]
    if sum(passage_totals) == len(passage_totals) * 10:  # 200
        print(f"{file.split('/')[-1]} is all clear")
    else:
        print(f"{file.split('/')[-1]} is not clear")
        print(sum(passage_totals))
        problem_idxs = [i for i in range(len(passage_totals)) if passage_totals[i] != 10]  # 200
        all_passages = [[retrieval_results[i]["ctxs"][j]["text"] for j in range(len(retrieval_results[i]["ctxs"]))] for i in problem_idxs]
        total_passages = [len(i) for i in all_passages]
        duplicates = [total_passages[i] - len(set(all_passages[i])) for i in range(len(all_passages))]
        empty_passages = [len([passage for passage in all_passages[i] if passage == ""]) for i in range(len(all_passages))]
        print(f"# Duplicates: {sum(duplicates)}\n# Empty passages: {sum(empty_passages)}")
        # if sum(duplicates):
        #     import ipdb; ipdb.set_trace()

        import IPython; IPython.embed()

# duplication happens in lie    s because there are duplicates in the retrieval index