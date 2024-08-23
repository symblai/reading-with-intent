from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, DistilBertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import json
from functools import partial

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8085'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


class sentiment_dataset(Dataset):
    def __init__(self, dataset_name, tokenizer, mode):
        super().__init__()
        self.tokenizer = tokenizer
        dataset = load_dataset('Blablablab/SOCKET', dataset_name, trust_remote_code=True)
        self.text = dataset[mode]["text"]
        self.labels = dataset[mode]["label"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.labels[idx]
        tokenized_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return {
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
            "label": torch.tensor(label)
        }

    def evaluate_results(self, predictions):
        predictions = torch.stack(predictions)
        labels = torch.tensor(self.labels)
        print(f"Overall Accuracy: {sum(predictions == labels) / len(labels) * 100:.2f}%")

    @staticmethod
    def collator_fn(batch, max_size):
        batch = [i for i in batch if i["input_ids"].shape[1] < max_size]
        max_length_inputs = max([i["input_ids"].shape[1] for i in batch])
        input_ids = torch.vstack([torch.nn.functional.pad(i["input_ids"], pad=(max_length_inputs - i["input_ids"].shape[1], 0)) for i in batch])
        attention_mask = torch.vstack([torch.nn.functional.pad(i["attention_mask"], pad=(max_length_inputs - i["attention_mask"].shape[1], 0)) for i in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.stack([i["label"] for i in batch]),
        }


class sarcasm_dataset(sentiment_dataset):
    def __init__(self, dataset_file, tokenizer):
        self.tokenizer = tokenizer
        dataset = json.load(open(dataset_file, "r"))
        self.text = [i["text"] for i in dataset]
        self.labels = [int(i["id"] > 1) for i in dataset]
        self.master_labels = [i["id"] for i in dataset]

    def evaluate_results(self, predictions):
        predictions = torch.stack(predictions)
        labels = torch.tensor(self.labels)
        master_labels = torch.tensor(self.master_labels)
        print(f"Overall Accuracy: {sum(predictions == labels)/len(labels)*100:.2f}%")
        print(f"Accuracy on sarcastic passages: {torch.sum((predictions == labels)[master_labels == 0]/torch.sum(master_labels == 0)*100):.2f}%")
        print(f"Accuracy on fact-distorted sarcastic passages: {torch.sum((predictions == labels)[master_labels == 1]/torch.sum(master_labels == 1)*100):.2f}%")
        print(f"Accuracy on fact-distorted passages: {torch.sum((predictions == labels)[master_labels == 2]/torch.sum(master_labels == 2)*100):.2f}%")
        print(f"Accuracy on original passages: {torch.sum((predictions == labels)[master_labels == 3]/torch.sum(master_labels == 3)*100):.2f}%")



def train_loop(rank, model, optimizer, dataloader):
    # torch.cuda.reset_peak_memory_stats()
    with tqdm(total=len(dataloader), position=rank) as pbar:
        for idx, batch in enumerate(dataloader):
            # print(torch.cuda.memory_summary())
            model.zero_grad(set_to_none=True)
            batch = {key: value.to(rank) for key, value in batch.items()}
            outputs = model.forward(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if idx % 600 == 0 and idx != 0:
                optimizer.param_groups[0]["lr"] *= 0.9
                optimizer.param_groups[1]["lr"] *= 0.9
                optimizer.param_groups[2]["lr"] *= 0.9
                optimizer.param_groups[3]["lr"] *= 0.7

            pbar.set_description(f"Loss: {loss.detach().item()}, LR1-6: {optimizer.param_groups[0]['lr']}, LR7: {optimizer.param_groups[2]['lr']}")

            # if idx % 30 == 0:
            #     torch.cuda.empty_cache()
            # print(torch.cuda.memory_summary())
            pbar.update(1)

def val_loop(rank, model, dataloader):
    results = []
    with tqdm(total=len(dataloader), position=rank) as pbar:
        for batch in dataloader:
            batch = {key: value.to(rank) for key, value in batch.items()}
            outputs = model.forward(**batch)
            results.extend(torch.argmax(outputs.logits.detach(), dim=1).to("cpu").detach())
            # acc_counter += torch.sum(torch.argmax(outputs.logits.detach(), dim=1) == batch["labels"].to(rank)).detach()
            pbar.update(1)
    return results
    # return torch.tensor([acc_counter], dtype=torch.float, device=rank)
    # print(f"Accuracy: {acc_counter/len(dataloader.dataset):}")


def main(rank, world_size):
    setup(rank, world_size)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    model_name = "/" + model_name
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    # model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base").cuda()


    dataset_name = "sarc"
    sarc_train = sentiment_dataset(dataset_name, tokenizer, "train")
    sarc_val = sentiment_dataset(dataset_name, tokenizer, "validation")
    sarc_val_synth = sarcasm_dataset("sarcasm_val_dataset.json", tokenizer)

    sampler_train = DistributedSampler(sarc_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    # sampler_val = DistributedSampler(sarc_val, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_collator = partial(sentiment_dataset.collator_fn, max_size=152)
    val_collator = partial(sentiment_dataset.collator_fn, max_size=512)
    sarc_dl_train = DataLoader(sarc_train, batch_size=75, sampler=sampler_train, num_workers=4, collate_fn=train_collator, pin_memory=True)
    sarc_dl_val = DataLoader(sarc_val, batch_size=100, shuffle=False, num_workers=4, drop_last=False, collate_fn=val_collator, pin_memory=True)
    sarc_dl_val_synth = DataLoader(sarc_val_synth, batch_size=50, shuffle=False, num_workers=4, drop_last=False, collate_fn=val_collator, pin_memory=True)


    optimizer = torch.optim.AdamW([
                                   {"params": model.module.roberta.encoder.layer[-3].parameters(), "lr": 5e-4},
                                   {"params": model.module.roberta.encoder.layer[-2].parameters(), "lr": 5e-4},
                                   {"params": model.module.roberta.encoder.layer[-1].parameters(), "lr": 5e-4},
                                   {"params": model.module.classifier.parameters(), "lr": 1e-3}])

    nepochs = 10

    for epoch in range(nepochs):
        model.eval()
        with torch.no_grad():
            if rank == 0:
                # sarc_dl_val.dataset.evaluate_results(val_loop(rank, model, sarc_dl_val))
                # if epoch % 3 == 0 and epoch != 0:
                sarc_dl_val_synth.dataset.evaluate_results(val_loop(rank, model, sarc_dl_val_synth))
                torch.save(model, f"{dataset_name}_{model_name.split('/')[1]}_classifier_epoch_{epoch}.pt")
        model.train()
        train_loop(rank, model, optimizer, sarc_dl_train)
    sarc_dl_val_synth.dataset.evaluate_results(val_loop(rank, model, sarc_dl_val_synth))
    torch.save(model, f"{dataset_name}_{model_name.split('/')[1]}_classifier_epoch_{epoch}.pt")
    cleanup()

# current_best = sarc_roberta-base_classifier_epoch_9.pt


if __name__ == '__main__':
    world_size = 2
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )
    # main(0, 1)