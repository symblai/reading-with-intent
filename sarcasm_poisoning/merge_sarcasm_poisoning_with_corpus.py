import pickle as pkl
from tqdm import tqdm


def clean_example(passage_idx, initial_passage, modified_passage):
    # initial_word = initial_passage.split(" ")[0].replace("\"", "")

    modified_passage = modified_passage.split("\n")
    modified_passage = [i for i in modified_passage if i]
    likely_start = [idx for idx, j in enumerate(modified_passage) if j[0] == "\""]
    if not likely_start:
        first_word = modified_passage[0].split(" ")
        if len(modified_passage) == 1:
            likely_start = [0]
        elif len(modified_passage) == 2 and "What" in first_word:
                likely_start = [0]
        elif "Oh" in first_word or "Oh," in first_word or "Wow" in first_word or "Wow," in first_word:
            likely_start = [0]
        elif "Here" in first_word or "Here's" in first_word or "I" in first_word or "I'll" in first_word:  # "What" in first_word or
            likely_start = [1]
        else:
            import ipdb; ipdb.set_trace()
    likely_start = likely_start[0]


    likely_end = [idx + 1 if (j[-1] == "\"") else idx for idx, j in enumerate(modified_passage) if (j[-1] == "\"" or
                  "note" in j.split(" ")[0].lower() or "distortions:" == j.split(" ")[0].lower() or
                  "i've" in j.split(" ")[0].lower() or "i" == j.split(" ")[0].lower() or "in this version" in " ".join(j.split(" ")[:3]).lower() or
                  "the original meaning" in " ".join(j.split(" ")[:3]).lower() or
                  "in this rewritten statement," in " ".join(j.split(" ")[:4]).lower() or
                  "in this rewritten version," in " ".join(j.split(" ")[:4]).lower() or
                  "in this version," in " ".join(j.split(" ")[:3]).lower() or
                  "in each case," in " ".join(j.split(" ")[:3]).lower() or
                  "in each sentence," in " ".join(j.split(" ")[:3]).lower() or
                  "in both cases" in " ".join(j.split(" ")[:3]).lower() or
                  "the sarcasm" in " ".join(j.split(" ")[:2]).lower() or
                  "please note" in " ".join(j.split(" ")[:2]).lower() or
                  "changes made:" in " ".join(j.split(" ")[:2]).lower() or
                  "distortions made:" in " ".join(j.split(" ")[:2]).lower() or
                  "distorted facts" in " ".join(j.split(" ")[:2]).lower() or
                  "notice how" in " ".join(j.split(" ")[:2]).lower() or
                  "the changes" in " ".join(j.split(" ")[:2]).lower() or
                  "changes i made" in " ".join(j.split(" ")[:3]).lower() or
                  "i tried to" in " ".join(j.split(" ")[:3]).lower() or
                  "i maintained the" in " ".join(j.split(" ")[:3]).lower() or
                  "i made sure to" in " ".join(j.split(" ")[:4]).lower() or
                  "i distorted" in " ".join(j.split(" ")[:2]).lower() or
                  "i aimed to" in " ".join(j.split(" ")[:3]).lower() or
                  "please let me" in " ".join(j.split(" ")[:3]).lower() or
                  "in the original" in " ".join(j.split(" ")[:3]).lower() or
                  "in the rewritten" in " ".join(j.split(" ")[:3]).lower() or
                  "in this rewritten" in " ".join(j.split(" ")[:3]).lower() or
                  "in this revised" in " ".join(j.split(" ")[:3]).lower() or
                  "the answer to" in " ".join(j.split(" ")[:3]).lower() or
                  "the answers to" in " ".join(j.split(" ")[:3]).lower() or
                  "the distortions i" in " ".join(j.split(" ")[:3]).lower() or
                  "let me know" in " ".join(j.split(" ")[:3]).lower() or
                  "the rest of" in " ".join(j.split(" ")[:3]).lower() or
                  "the goal is" in " ".join(j.split(" ")[:3]).lower() or
                  "this rewritten statement" in " ".join(j.split(" ")[:3]).lower() or
                  "this rewritten version" in " ".join(j.split(" ")[:3]).lower() or
                  "this rewritten passage" in " ".join(j.split(" ")[:3]).lower() or
                  "the denotation of" in " ".join(j.split(" ")[:3]).lower() or
                  "the denotation (the" in " ".join(j.split(" ")[:3]).lower() or
                  "the rewritten statement" in " ".join(j.split(" ")[:3]).lower() or
                  "the rewritten text" in " ".join(j.split(" ")[:3]).lower() or
                  "the rewritten passage" in " ".join(j.split(" ")[:3]).lower() or
                  "in this rewritten" in " ".join(j.split(" ")[:3]).lower() or
                  "the connotation remains" in " ".join(j.split(" ")[:3]).lower() or
                  "the subtle changes" in " ".join(j.split(" ")[:3]).lower() or
                  "the original text" in " ".join(j.split(" ")[:3]).lower() or
                  "the original sentence" in " ".join(j.split(" ")[:3]).lower() or
                  "the original statement" in " ".join(j.split(" ")[:3]).lower() or
                  "the original phrase" in " ".join(j.split(" ")[:3]).lower() or
                  "the original passage" in " ".join(j.split(" ")[:3]).lower() or
                  "the original facts" in " ".join(j.split(" ")[:3]).lower() or
                  "the original answers" in " ".join(j.split(" ")[:3]).lower() or
                  "the original answer" in " ".join(j.split(" ")[:3]).lower() or
                  "the passage still" in " ".join(j.split(" ")[:3]).lower() or
                  "the sarcastic tone" in " ".join(j.split(" ")[:3]).lower() or
                  "the denotation has" in " ".join(j.split(" ")[:3]).lower() or
                  "the tone is" in " ".join(j.split(" ")[:3]).lower() or
                  "the denotative meaning" in " ".join(j.split(" ")[:3]).lower() or
                  "the connotative meaning" in " ".join(j.split(" ")[:3]).lower() or
                  "the distorted facts" in " ".join(j.split(" ")[:3]).lower() or
                  "the distortions made" in " ".join(j.split(" ")[:3]).lower() or
                  "the connotation of" in " ".join(j.split(" ")[:3]).lower() or
                  "the overall meaning" in " ".join(j.split(" ")[:3]).lower() or
                  "the overall connotation" in " ".join(j.split(" ")[:3]).lower() or
                  "the overall connotative" in " ".join(j.split(" ")[:3]).lower() or
                  "these changes are" in " ".join(j.split(" ")[:3]).lower() or
                  "here's what i" in " ".join(j.split(" ")[:3]).lower() or
                  "here are the" in " ".join(j.split(" ")[:3]).lower() or
                  "here, i've made" in " ".join(j.split(" ")[:3]).lower() or
                  "here, i've maintained" in " ".join(j.split(" ")[:3]).lower() or
                  "here, i've changed" in " ".join(j.split(" ")[:3]).lower() or
                  "here, i've distorted" in " ".join(j.split(" ")[:3]).lower() or
                  "here, i've kept" in " ".join(j.split(" ")[:3]).lower() or
                  "here, i distorted" in " ".join(j.split(" ")[:3]).lower() or
                  "here, i've replaced" in " ".join(j.split(" ")[:3]).lower() or
                  "here, i made" in " ".join(j.split(" ")[:3]).lower() or
                  "here, i changed" in " ".join(j.split(" ")[:3]).lower() or
                  "here's a breakdown" in " ".join(j.split(" ")[:3]).lower() or
                  "here, i've subtly distorted" in " ".join(j.split(" ")[:4]).lower() or
                  "here are the distortions" in " ".join(j.split(" ")[:4]).lower() or
                  "here are the specific distortions" in " ".join(j.split(" ")[:5]).lower() or
                  "the possible answers to" in " ".join(j.split(" ")[:4]).lower() or
                  "let me know if" in " ".join(j.split(" ")[:4]).lower() or
                  "in this rewritten text," in " ".join(j.split(" ")[:4]).lower() or
                  "in this rewritten passage," in " ".join(j.split(" ")[:4]).lower())
                  and idx > likely_start]
    if not likely_end and likely_start == len(modified_passage)-1:
        likely_end = [len(modified_passage)]
    if not likely_end:
        if likely_start == 0:
            likely_end = [-1]
        else:
            import ipdb; ipdb.set_trace()
    # if passage_idx == 115 or passage_idx == 149 or passage_idx == 643:
    #     import ipdb; ipdb.set_trace()
    likely_end = likely_end[0]
    return "\n".join(modified_passage[likely_start:likely_end])[1:-1]


if __name__ == "__main__":
    retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage.pkl", "rb"))

    # Merge Top-200 Sarcasm with Dataset
    with open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b.pkl", "rb") as file:
        gpl_sarcasm = pkl.load(file)
    with open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_1000_1600.pkl", "rb") as file:
        gpl_sarcasm.extend(pkl.load(file))
    with open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_1600_2200.pkl", "rb") as file:
        gpl_sarcasm.extend(pkl.load(file))
    with open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_2200_3400.pkl", "rb") as file:
        gpl_sarcasm.extend(pkl.load(file))
    with open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_3400_3900.pkl", "rb") as file:
        gpl_sarcasm.extend(pkl.load(file))
    with open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_3900_4500.pkl", "rb") as file:
        gpl_sarcasm.extend(pkl.load(file))
    with open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_4500_5100.pkl", "rb") as file:
        gpl_sarcasm.extend(pkl.load(file))
    with open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_5100_5350.pkl", "rb") as file:
        gpl_sarcasm.extend(pkl.load(file))
    with open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_5350_6600.pkl", "rb") as file:
        gpl_sarcasm.extend(pkl.load(file))

    gpl_sarcasmed = [i.outputs[0].text for i in gpl_sarcasm]
    with tqdm(total=len(gpl_sarcasmed)) as pbar:
        for i in range(len(gpl_sarcasmed)):
            # if i == 35519:
            #     import ipdb; ipdb.set_trace()
            gpl_sarcasmed[i] = clean_example(i, retrieval_results[i//200]["ctxs"][i%200], gpl_sarcasmed[i])
            pbar.update(1)

    gpl_sarcasmed = [gpl_sarcasmed[i:i+200] for i in range(0, len(gpl_sarcasmed), 200)]
    for i in range(len(retrieval_results)):
        for j in range(len(retrieval_results[i]["ctxs"])):
            retrieval_results[i]["ctxs"][j]["text"] = gpl_sarcasmed[i][j]
    pkl.dump(retrieval_results, open("../retrieval/gpl_retrieval_results_w_passage_sarcastic_fullv3.pkl", "wb"))



    ## Merge Top-10 Sarcasm with Dataset
    retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage.pkl", "rb"))[:1000]
    gpl_sarcasm = pkl.load(open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b.pkl", "rb"))
    gpl_sarcasm = [j for i in range(0, len(gpl_sarcasm), 200) for j in gpl_sarcasm[i:i+10]]
    gpl_sarcasm2 = pkl.load(open("gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_0_1000_10.pkl", "rb"))

    gpl_sarcasmed = [[gpl_sarcasm2[i].outputs[0].text, gpl_sarcasm2[i].outputs[1].text, gpl_sarcasm[i].outputs[0].text] for i in range(len(gpl_sarcasm))]
    for i in range(len(gpl_sarcasmed)):
        for j in range(len(gpl_sarcasmed[i])):
            passage = gpl_sarcasmed[i][j].split("\n")
            passage = [i for i in passage if i and (i[0] == "\"" or i[-1] == "\"")]
            gpl_sarcasmed[i][j] = "\n".join(passage)[1:-1]
    new_sarcasmed = [[j for j in i if j[:2] != "Oh"] for i in gpl_sarcasmed]
    gpl_sarcasmed = [i[0] if i else "" for i in new_sarcasmed]
    gpl_sarcasmed = [gpl_sarcasmed[i:i+10] for i in range(0, len(gpl_sarcasmed), 10)]
    for i in range(len(retrieval_results)):
        for j in range(len(gpl_sarcasmed[i])):
            if gpl_sarcasmed[i][j]:
                retrieval_results[i]["ctxs"][j]["text"] = gpl_sarcasmed[i][j]
            else:
                del retrieval_results[i]["ctxs"][j]
        retrieval_results[i]["ctxs"] = retrieval_results[i]["ctxs"][:10]
    pkl.dump(retrieval_results, open("../retrieval/gpl_retrieval_results_w_passage_sarcastic_1000_no_oh.pkl", "wb"))



    ### Merge Fact Distorted with Dataset

    with open("gpl_retrieval_results_fact_distorted_llama3_70b.pkl", "rb") as f:
        gpl_lies = pkl.load(f)
    with open("gpl_retrieval_results_fact_distorted_llama3_70b_1000_2000.pkl", "rb") as f:
        gpl_lies.extend(pkl.load(f))
    with open("gpl_retrieval_results_fact_distorted_llama3_70b_2000_2600.pkl", "rb") as f:
        gpl_lies.extend(pkl.load(f))
    with open("gpl_retrieval_results_fact_distorted_llama3_70b_2600_3400.pkl", "rb") as f:
        gpl_lies.extend(pkl.load(f))
    with open("gpl_retrieval_results_fact_distorted_llama3_70b_3400_3800.pkl", "rb") as f:
        gpl_lies.extend(pkl.load(f))
    with open("gpl_retrieval_results_fact_distorted_llama3_70b_3800_5000.pkl", "rb") as f:
        gpl_lies.extend(pkl.load(f))
    with open("gpl_retrieval_results_fact_distorted_llama3_70b_5000_6600.pkl", "rb") as f:
        gpl_lies.extend(pkl.load(f))

    gpl_lied = [i.outputs[0].text for i in gpl_lies]
    with tqdm(total=len(gpl_lied)) as pbar:
        for i in range(len(gpl_lied)):
            gpl_lied[i] = clean_example(i, retrieval_results[i//200]["ctxs"][i%200], gpl_lied[i])
            pbar.update(1)

    gpl_lied = [gpl_lied[i:i+200] for i in range(0, len(gpl_lied), 200)]
    for i in range(len(retrieval_results)):
        for j in range(len(retrieval_results[i]["ctxs"])):
            retrieval_results[i]["ctxs"][j]["text"] = gpl_lied[i][j]
    pkl.dump(retrieval_results, open("../retrieval/gpl_retrieval_results_w_passage_liesv3.pkl", "wb"))


    ## Merge Sarcastic Fact Distorted with Dataset
    with open("gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_0_1600.pkl", "rb") as file:
        gpl_sarcastic_lies = pkl.load(file)
    with open("gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_1600_2300.pkl", "rb") as file:
        gpl_sarcastic_lies.extend(pkl.load(file))
    with open("gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_2300_2700.pkl", "rb") as file:
        gpl_sarcastic_lies.extend(pkl.load(file))
    with open("gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_2700_3300.pkl", "rb") as file:
        gpl_sarcastic_lies.extend(pkl.load(file))
    with open("gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_3300_4400.pkl", "rb") as file:
        gpl_sarcastic_lies.extend(pkl.load(file))
    with open("gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_4400_5400.pkl", "rb") as file:
        gpl_sarcastic_lies.extend(pkl.load(file))
    with open("gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_5400_6700.pkl", "rb") as file:
        gpl_sarcastic_lies.extend(pkl.load(file))

    gpl_lied = [i.outputs[0].text for i in gpl_sarcastic_lies]
    with tqdm(total=len(gpl_lied)) as pbar:
        for i in range(len(gpl_lied)):
            gpl_lied[i] = clean_example(i, retrieval_results[i//200]["ctxs"][i%200], gpl_lied[i])
            pbar.update(1)

    import ipdb; ipdb.set_trace()

    gpl_lied = [gpl_lied[i:i+200] for i in range(0, len(gpl_lied), 200)]
    for i in range(len(retrieval_results)):
        for j in range(len(retrieval_results[i]["ctxs"])):
            retrieval_results[i]["ctxs"][j]["text"] = gpl_lied[i][j]
    pkl.dump(retrieval_results, open("../retrieval/gpl_retrieval_results_w_passage_sarcastic_lies.pkl", "wb"))
