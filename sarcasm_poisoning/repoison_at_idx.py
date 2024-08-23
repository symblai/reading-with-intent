from vllm import LLM, SamplingParams
import pickle as pkl
import json
import numpy as np
import os
import huggingface_hub
from transformers import AutoTokenizer
import torch
import ray

file_translate_sarcasm = [
    [[0, 1000], "gpl_retrieval_results_sarcasmed_prompt2_llama3_70b.pkl"],
    [[1000, 1600], "gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_1000_1600.pkl"],
    [[1600, 2200], "gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_1600_2200.pkl"],
    [[2200, 3400], "gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_2200_3400.pkl"],
    [[3400, 3900], "gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_3400_3900.pkl"],
    [[3900, 4500], "gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_3900_4500.pkl"],
    [[4500, 5100], "gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_4500_5100.pkl"],
    [[5100, 5350], "gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_5100_5350.pkl"],
    [[5350, 6600], "gpl_retrieval_results_sarcasmed_prompt2_llama3_70b_5350_6600.pkl"],
]

file_translate_lies = [
    [[0, 1000], "gpl_retrieval_results_fact_distorted_llama3_70b.pkl"],
    [[1000, 2000], "gpl_retrieval_results_fact_distorted_llama3_70b_1000_2000.pkl"],
    [[2000, 2600], "gpl_retrieval_results_fact_distorted_llama3_70b_2000_2600.pkl"],
    [[2600, 3400], "gpl_retrieval_results_fact_distorted_llama3_70b_2600_3400.pkl"],
    [[3400, 3800], "gpl_retrieval_results_fact_distorted_llama3_70b_3400_3800.pkl"],
    [[3800, 5000], "gpl_retrieval_results_fact_distorted_llama3_70b_3800_5000.pkl"],
    [[5000, 6600], "gpl_retrieval_results_fact_distorted_llama3_70b_5000_6600.pkl"],
]


file_translate_sarcastic_lies = [
    [[0, 1600], "gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_0_1600.pkl"],
    [[1600, 2300], "gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_1600_2300.pkl"],
    [[2300, 2700], "gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_2300_2700.pkl"],
    [[2700, 3300], "gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_2700_3300.pkl"],
    [[3300, 4400], "gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_3300_4400.pkl"],
    [[4400, 5400], "gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_4400_5400.pkl"],
    [[5400, 6700], "gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_5400_6700.pkl"],
]


def get_passages_to_sub(filename):
    file_to_edit = open(filename, "rb")
    passages_to_sub = pkl.load(file_to_edit)
    file_to_edit.close()
    return passages_to_sub


def main():
    hf_token = os.environ["HF_KEY"]
    huggingface_hub.login(hf_token)

    start = 0
    end = 6700
    sarcastic = True
    answer_agree = False

    retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passages_fact_distorted_v3.pkl", "rb"))[start:end]
    retrieval_passages = [{"passage": j, "question": i["question"], "answer": i["answers"]} for i in retrieval_results for j in i["ctxs"]]
    repoisoned_idx = [idx*200+idx2 for idx, i in enumerate(retrieval_results) for idx2, j in enumerate(i["ctxs"]) if j["repoison"]]

    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side="left", trust_remote_code=True)

    # Sarcasm creation prompt
    if sarcastic:
        prompt = ("Sarcasm is when you write or say one thing but mean the opposite. This clear through changing the writing patterns and style. "
              "It changes what you write denotatively without changing it connotatively. "
              "It is a covertly deceptive way to communicate. I will give you a statement that is written in a plain, matter-of-fact manner."
              "I want you to convert it to be sarcastic. The overall meaning connotatively should stay the same, but the denotation should be different. "
              "Please do not make the sarcasm over the top. It should be subtle. ")
    else:
        prompt = ("I will give you a passage. It will contain numerous facts. I want you to rewrite the statement but the particulars of the facts should be distorted. "
              "Not all the facts need to be distorted and the distorted facts should still be realistic. Do not invent fake things (broadly defined) to distory the facts. "
              "The distortion should be subtle and not over the top."
              "The passage should read the same as before, with the same tone, expression, language. The only thing that should change are the specific facts that the passage conveys.")

    inputs = [tokenizer.apply_chat_template([{"role": "user", "content": f"{prompt} When rewriting this passage  "
             f"{'to be sarcastic' if sarcastic else 'to distort the facts'} make sure that any of the possible answers in the passage to the question \'{passage['question']}\'" +
                                                                         (f": '{' '.join(passage['answer'])}' " if answer_agree and passage['passage']['repoison'] else "") +
             f" {'is still' if sarcastic else 'is no longer'} in the passage."
             f"\nPassage: {passage['passage']['text']}"}], tokenize=False, add_generation_prompt=True) for passage in retrieval_passages]

    # ray.init(logging_level='ERROR')
    sampling_params = SamplingParams(temperature=0.5, top_p=1, max_tokens=1024)
    model = LLM(model=model_id, tensor_parallel_size=4, trust_remote_code=True)
    while True:
        idx = input("What index would you like to modify? ")
        if idx == "q":
            break
        elif isinstance(idx, str) and os.path.exists(idx):
            indices = np.array(pkl.load(open(idx, "rb")))
            model_inputs = [inputs[i] for i in indices]
        else:
            model_inputs = inputs[int(idx)]
            indices = [int(idx)]
        with torch.no_grad():
            results = model.generate(model_inputs, sampling_params)

        if len(indices) == 1:
            print(results[0].outputs[0].text)

        edit_file_range_file = [[i for i in file_translate_sarcastic_lies if idx // 200 in range(i[0][0], i[0][1])][0] for idx in indices]
        file_ranges = [i[0] for i in edit_file_range_file]
        edit_file = [i[1] for i in edit_file_range_file]
        file_indices = [indices[i] - file_ranges[i][0]*200 for i in range(len(indices))]
        file_change = [0] + np.argwhere(~np.array([True] + [edit_file[i]==edit_file[i-1] for i in range(1, len(edit_file))])).reshape(-1).tolist()
        cur_file = 0
        passages_to_sub = get_passages_to_sub(edit_file[file_change[cur_file]])
        # repoisoned_passages_to_sub = get_passages_to_sub(f"gpl_retrieval_results_fact_distorted_prompt2_llama3_70b_0_6700_repoisoned.pkl")
        for j, idx in enumerate(indices):
            if j in file_change and j != 0:
                with open(edit_file[file_change[cur_file]], 'wb') as f:
                    pkl.dump(passages_to_sub, f)
                cur_file += 1
                passages_to_sub = get_passages_to_sub(edit_file[file_change[cur_file]])
            results[j].request_id = passages_to_sub[file_indices[j]].request_id
            passages_to_sub[file_indices[j]] = results[j]

            # if idx in repoisoned_idx:
            #     results[j].request_id = repoisoned_passages_to_sub[repoisoned_idx.index(idx)].request_id
            #     repoisoned_passages_to_sub[repoisoned_idx.index(idx)] = results[j]
        with open(edit_file[file_change[cur_file]], 'wb') as f:
            pkl.dump(passages_to_sub, f)
        # with open(f"gpl_retrieval_results_fact_distorted_prompt2_llama3_70b_0_6700_repoisoned.pkl", 'wb') as f:
        #     pkl.dump(repoisoned_passages_to_sub, f)



if __name__ == "__main__":
    main()