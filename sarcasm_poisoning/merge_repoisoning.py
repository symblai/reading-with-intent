import pickle as pkl
from merge_sarcasm_poisoning_with_corpus import clean_example
from tqdm import tqdm

retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage.pkl", "rb"))
original_sarcastic = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage_liesv3.pkl", "rb"))
repoisoned_results = pkl.load(open("gpl_retrieval_results_fact_distorted_prompt2_llama3_70b_0_6700_repoisoned.pkl", "rb"))


repoisoned_results = [i.outputs[0].text for i in repoisoned_results]

k = 0
with tqdm(total=len(retrieval_results)) as pbar:
    for i in range(len(retrieval_results)):
        for j in range(len(retrieval_results[i]["ctxs"])):
            if retrieval_results[i]["ctxs"][j]["repoison"]:
                repoisoned_results[k] = clean_example(k, retrieval_results[i]["ctxs"][j], repoisoned_results[k])
                retrieval_results[i]["ctxs"][j]["text"] = repoisoned_results[k]
                k += 1
            else:
                retrieval_results[i]["ctxs"][j]["text"] = original_sarcastic[i]["ctxs"][j]["text"]
        pbar.update(1)

pkl.dump(retrieval_results, open("../retrieval/gpl_retrieval_results_w_passages_fact_distorted_v3.pkl", "wb"))