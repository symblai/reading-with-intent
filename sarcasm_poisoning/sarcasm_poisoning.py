from vllm import LLM, SamplingParams
import pickle as pkl
import json
import numpy as np
import os
import huggingface_hub
from transformers import AutoTokenizer
import torch
import ray


def main():
    hf_token = os.environ["HF_KEY"]
    huggingface_hub.login(hf_token)

    start = 0
    end = 6700

    retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage.pkl", "rb"))[start:end]
    retrieval_passages = [j for i in retrieval_results for j in i["ctxs"]]



    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side="left", trust_remote_code=True)

    prompt = ("Sarcasm is when you write or say one thing but mean the opposite. This clear through changing the writing patterns and style. "
              "It changes what you write denotatively without changing it connotatively. "
              "It is a covertly deceptive way to communicate. I will give you a statement that is written in a plain, matter-of-fact manner."
              "I want you to convert it to be sarcastic. The overall meaning connotatively should stay the same, but the denotation should be different. "
              "Please do not make the sarcasm over the top. It should be subtle.")

    # prompt = ("I will give you a passage. It will contain numerous facts. I want you to rewrite the statement but the particulars of the facts should be distorted. "
    #           "Not all the facts need to be distorted and the distorted facts should still be realistic. Do not invent fake things (broadly defined) to distory the facts. "
    #           "The distortion should be subtle and not over the top."
    #           "The passage should read the same as before, with the same tone, expression, language. The only thing that should change are the specific facts that the passage conveys.")

    # inputs = [tokenizer.apply_chat_template([{"role": "system", "content": prompt},
    #            {"role": "user", "content": f"{passage}"}], tokenize=False, add_generation_prompt=True) for passage in retrieval_passages]
    inputs = [tokenizer.apply_chat_template([{"role": "user", "content": f"{prompt}\n{passage['text']}"}], tokenize=False, add_generation_prompt=True) for passage in retrieval_passages]

    # ray.init(logging_level='ERROR')
    sampling_params = SamplingParams(temperature=0.5, top_p=1, max_tokens=1024)
    model = LLM(model=model_id, tensor_parallel_size=4, trust_remote_code=True)
    with torch.no_grad():
        results = model.generate(inputs, sampling_params)

    # pkl.dump(results, open(f"gpl_retrieval_results_fact_distorted_llama3_70b_{start}_{end}.pkl", "wb"))
    pkl.dump(results, open(f"gpl_retrieval_results_fact_distorted_sarcasmed_prompt2_llama3_70b_{start}_{end}.pkl", "wb"))

    import IPython; IPython.embed()

if __name__ == "__main__":
    main()