import pickle as pkl
from emf1_eval import get_em_f1



def get_acc(llm_answers, retrieval_results):
    inferred_answers = [i.outputs[0].text.strip() for i in llm_answers]
    gt_answers = [i["answers"] for i in retrieval_results]
    em, f1 = get_em_f1(retrieval_results, inferred_answers)
    num_correct = [sum([gt_answers[i][j].lower().strip() in inferred_answers[i].lower().strip() for j in range(len(gt_answers[i]))]) > 0 for i in range(len(inferred_answers))]
    print(f"Accuracy: {sum(num_correct) / len(gt_answers)}")


model_names = [
    "llama2-7b-chat",
    "llama2-70b-chat"
    "mistral-7b",
    "mixtral-8x22b",
    "phi-3-mini",
    "phi-3-small",
    "phi-3-medium",
    "qwen2-0.5b", "qwen2-1.5b",
    "qwen2-7b",
    "qwen2-72b"
]

dataset_name = ["base_ds", "fully_sarcastic", "sarcasm_w_distortion_manual", "sarcasm_w_distortion_retrieved"]
intent_positions = ["before", "after"]
intent_labels = ["Language Tone: "]
intent_categories = [["Sarcastic", "Straightforward"]]
promptiness = ["base_prompt", "full_prompt"]
retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage.pkl", "rb"))
intent_tags = [False, True]
file_paths = []

if not file_paths:
    for model_name in model_names:
        for ds_name in dataset_name:
            for prompted in promptiness:
                for intent_tag in intent_tags:
                    for intent_position in intent_positions:
                        for k in range(len(intent_labels)):
                            for l in range(len(intent_categories)):
                                file_path = f"results/{model_name}_nq_answers_gpl_{dataset_name}_prefix_retrieved{f'_intent_prompt' if prompted == 'full_prompt' else 'base_prompt'}{f'_intent_tag_{k}_{l}_{intent_position}' if intent_tag else '_no_intent_tag'}_temp_0.pkl"
                                llm_answers = pkl.load(open(file_path, "rb"))[0]
                                print(f"Model Name: {model_name} | Dataset Name: {ds_name} | Prompt: {'Base Prompt' if prompted == 'base_prompt' else 'Full Prompt'} | Intent Tag: {intent_tag} | Intent Position: {intent_position}")
                                get_acc(llm_answers, retrieval_results)
                print()
if file_paths:
    for file_path in file_paths:
        llm_answers = pkl.load(open(file_path, "rb"))[0]
        print(file_path)
        get_acc(llm_answers, retrieval_results)
        print()
# import IPython; IPython.embed()
