from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch
from collections import OrderedDict
import os
import torch.distributed as dist
import pickle as pkl
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8085'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


class passage_dataset(Dataset):
    def __init__(self, dataset_file, tokenizer):
        self.data = pkl.load(open(dataset_file, 'rb'))
        self.flattened_data = [j["text"] for i in self.data for j in i["ctxs"]]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.flattened_data)

    def __getitem__(self, idx):
        passage = self.flattened_data[idx]
        tokenized_passage = self.tokenizer(passage, return_tensors="pt", truncation=True)
        return {
            "input_ids": tokenized_passage["input_ids"],
            "attention_mask": tokenized_passage["attention_mask"],
            "idx": idx
        }

    def __setitem__(self, idx, value):
        self.data[idx//10]["ctxs"][idx%10]["pred"] = value

    def save(self, file_path):
        pkl.dump(self.data, open(file_path, "wb"))

    @staticmethod
    def collate_fn(batch):
        max_length_inputs = max([i["input_ids"].shape[1] for i in batch])
        input_ids = torch.vstack([torch.nn.functional.pad(i["input_ids"], pad=(max_length_inputs - i["input_ids"].shape[1], 0)) for i in batch])
        attention_mask = torch.vstack([torch.nn.functional.pad(i["attention_mask"], pad=(max_length_inputs - i["attention_mask"].shape[1], 0)) for i in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "idx": torch.tensor([i["idx"] for i in batch]),
        }


def main(rank, worldsize):
    setup(rank, worldsize)
    model_name = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).cuda()

    # Load the state_dict without initializing the process group
    state_dict = torch.load("sarc_roberta-base_classifier_epoch_9.pt", map_location="cpu")

    # Remove DDP prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.state_dict().items():
        new_key = k.replace("module.", "")  # remove 'module.' prefix if present
        new_state_dict[new_key] = v

    # Load the modified state_dict into the model
    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    model.eval()

    dataset_file_names = [
        # "../../retrieval/gpl_retrieval_results_w_passage.pkl",
        # "../../retrieval/gpl_retrieval_results_w_passages_fully_sarcastic_v3.pkl",
        "../../sarcasm_poisoning/20p_sarcastic_20p_fact_distorted_prefix_sarcastic_poisoned_retrieval_corpus.pkl",
        # "../../retrieval/gpl_retrieval_results_lying_sarcasm_in_corpus_w_passage.pkl"
    ]
    passage_datasets = [passage_dataset(dataset_file_names[i], tokenizer) for i in range(len(dataset_file_names))]
    passage_dataloaders = [DataLoader(passage_datasets[i], batch_size=150, shuffle=False, num_workers=4, collate_fn=passage_dataset.collate_fn, pin_memory=True) for i in range(len(passage_datasets))]

    for k, passage_dataloader in enumerate(passage_dataloaders):
        with tqdm(total=len(passage_dataloader)) as pbar:
            for batch in passage_dataloader:
                with torch.no_grad():
                    idxes = batch["idx"]
                    del batch["idx"]
                    batch = {key: value.cuda() for key, value in batch.items()}
                    output = model(**batch)
                    results = torch.argmax(output.logits, dim=1) == 0
                    for i in range(len(results)):
                        passage_dataloader.dataset[idxes[i]] = results[i].cpu().detach().item()
                    pbar.update(1)
            passage_dataloader.dataset.save(dataset_file_names[k].split("/")[-1][:-4]+"_pred_intent.pkl")



if __name__ == "__main__":
    main(0, 1)
