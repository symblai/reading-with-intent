from vllm import LLM, SamplingParams
import pickle as pkl
import json
import numpy as np
import os
import huggingface_hub
from transformers import AutoTokenizer
import torch
import random
from datetime import datetime
import ray


def run_model(model, tokenizer, prompt, q_p_pair, temp):
    inputs = [tokenizer.apply_chat_template([{"role": "system", "content": prompt}, {"role": "user", "content": f"{passage}"}], tokenize=False, add_generation_prompt=True) for passage in q_p_pair]

    sampling_params = SamplingParams(temperature=temp, top_p=1, max_tokens=1024)
    with torch.no_grad():
        results = model.generate(inputs, sampling_params)
    return results


def main(model_idx):
    ray.init(logging_level='ERROR')
    hf_token = os.environ["HF_KEY"]
    huggingface_hub.login(hf_token)

    model_ids = [
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3-mini-128k-instruct",
        "microsoft/Phi-3-small-128k-instruct",
        "Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "microsoft/Phi-3-medium-128k-instruct",
        "meta-llama/Llama-2-70b-chat-hf",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "Qwen/Qwen2-72B-Instruct"
    ]
    model_name = [
        "llama2-7b-chat",
        "mistral-7b",
        "phi-3-mini",
        "phi-3-small",
        "qwen2-0.5b",
        "qwen2-1.5b",
        "qwen2-7b",
        "phi-3-medium",
        "llama2-70b-chat",
        "mixtral-8x22b",
        "qwen2-72b",
    ]
    promptiness = [
        "base_prompt",
        "full_prompt"
    ]
    datasets = [
        ("../retrieval/gpl_retrieval_results_w_passage.pkl", "base_ds", lambda _: 0),
        ("../retrieval/gpl_retrieval_results_w_passages_fully_sarcastic_v3.pkl", "fully_sarcastic", lambda _: 1),
        ("../sarcasm_poisoning/20p_sarcastic_20p_fact_distorted_prefix_sarcastic_poisoned_retrieval_corpus.pkl", "sarcasm_w_distortion_manual", lambda x: int(x['sarcastic'])),
        ("../sarcasm_poisoning/20p_sarcastic_20p_fact_distorted_postfix_sarcastic_poisoned_retrieval_corpus.pkl", "sarcasm_w_distortion_manual_postfix", lambda x: int(x['sarcastic'])),
        ("../retrieval/gpl_retrieval_results_lying_sarcasm_in_corpus_w_passage.pkl", "sarcasm_w_distortion_retrieved", lambda x: x['id'] > 21015324),
        ("sentiment_classifier/gpl_retrieval_results_w_passage_pred_intent.pkl", "nonoracle_base_ds", lambda doc: doc['pred']),
        ("sentiment_classifier/gpl_retrieval_results_w_passages_fully_sarcastic_v3_pred_intent.pkl", "nonoracle_fully_sarcastic", lambda doc: doc['pred']),
        ("sentiment_classifier/20p_sarcastic_20p_fact_distorted_prefix_sarcastic_poisoned_retrieval_corpus_pred_intent.pkl", "nonoracle_sarcasm_w_distortion_manual", lambda doc: doc['pred']),
        ("sentiment_classifier/gpl_retrieval_results_lying_sarcasm_in_corpus_w_passage_pred_intent.pkl", "nonoracle_sarcasm_w_distortion_retrieved", lambda doc: doc['pred'])
    ]

    intent_tags = [False, True]
    intent_positions = ["before", "after"]
    intent_labels = ["Language Tone: "]
    intent_categories = [["Sarcastic", "Straightforward"]]
    f1 = [False, True]

    intent_prompt = [
        "Given the potential for emotionally charged language in these internet search results, ensure your response fully and accurately conveys both the denotative and connotative meanings.",
    ]
    print(model_name[model_idx])
    model = LLM(model=model_ids[model_idx], tensor_parallel_size=2 if 'phi' in model_name[model_idx] else 8, trust_remote_code=True, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_ids[model_idx], token=hf_token, padding_side="left", trust_remote_code=True)
    for ds_name_idx, dataset_file_name in enumerate(datasets):
        dataset_file_name = datasets[ds_name_idx][0]
        dataset_name = datasets[ds_name_idx][1]
        dataset_to_sarc = datasets[ds_name_idx][2]
        retrieval_results = pkl.load(open(dataset_file_name, "rb"))
        for prompted in promptiness:
            prompt = ("Write a high-quality answer for the given question using only your knowledge of the question and the provided search results (some of which might be irrelevant). " +
                      (intent_prompt[0] if "full_prompt" in prompted else "") +
                      "The answer should only contain 1-3 words.")
            for intent_tag in intent_tags:
                if intent_tag:
                    for intent_position in intent_positions:
                        k = 0
                        l = 0
                        if intent_position == "after":
                            q_p_pair = ["\n".join([f"Document [{i}] (Title:{doc['title']}) {doc['text']}\n{intent_labels[k]}{intent_categories[l][0] if dataset_to_sarc(doc) else intent_categories[l][1]}" for i, doc in enumerate(retrieval_results[j]["ctxs"][:10])]) + f"\n\nQuestion: {retrieval_results[j]['question']}\n\n{'The answer should not exceed 3 words.' if f1 else ''}\n\nAnswer: " for j in range(len(retrieval_results))]
                        elif intent_position == "before":
                            q_p_pair = ["\n".join([f"{intent_labels[k]}{intent_categories[l][0] if dataset_to_sarc(doc) else intent_categories[l][1]}\nDocument [{i}] (Title:{doc['title']}) {doc['text']}" for i, doc in enumerate(retrieval_results[j]["ctxs"][:10])]) + f"\n\nQuestion: {retrieval_results[j]['question']}\n\n{'The answer should not exceed 3 words.' if f1 else ''}\n\nAnswer: " for j in range(len(retrieval_results))]
                        # file_path = f"llama2_{size}b_nq_answers_gpl_{dataset_name[ds_name_idx]}_prefix_retrieved{'_intent_prompt' if intent_prompt else ''}_intent_tag_{k}_{intent_position}_temp_0.pkl"
                        file_path = f"results/{model_name[model_idx]}_nq_answers_gpl_{dataset_name}_prefix_retrieved{f'_intent_prompt' if prompted == 'full_prompt' else 'base_prompt'}{f'_intent_tag_{k}_{l}_{intent_position}' if intent_tag else '_no_intent_tag'}_temp_0.pkl"
                        if not os.path.exists(file_path):
                            results = run_model(model, tokenizer, prompt, q_p_pair, temp=0)
                            pkl.dump([results], open(file_path, "wb"))
                        else:
                            print(file_path)
                            print("skipped")
                else:
                    q_p_pair = ["\n".join([f"Document [{i}] (Title:{doc['title']}) {doc['text']}" for i, doc in enumerate(retrieval_results[j]["ctxs"][:10])]) + f"\n\nQuestion: {retrieval_results[j]['question']}\n\n{'The answer should not exceed 3 words.' if f1 else ''}\n\nAnswer: " for j in range(len(retrieval_results))]
                    # file_path = f"llama2_{size}b_nq_answers_gpl_{dataset_name[ds_name_idx]}_prefix_retrieved{'_intent_prompt' if intent_prompt else ''}_temp_0.pkl"
                    file_path = f"results/{model_name[model_idx]}_nq_answers_gpl_{dataset_name}_prefix_retrieved_{f'_intent_prompt' if prompted == 'full_prompt' else prompted}{'_intent_tag' if intent_tag else '_no_intent_tag'}_temp_0.pkl"
                    if not os.path.exists(file_path):
                        results = run_model(model, tokenizer, prompt, q_p_pair, temp=0)
                        pkl.dump([results], open(file_path, "wb"))
                    else:
                        print(file_path)
                        print("skipped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id")
    args = parser.parse_args()
    main(int(args.model_id))
