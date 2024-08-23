### Code from a previous project

from transformers import AutoTokenizer, AutoModel
import torch
import tqdm
import os
import torch
import torch.distributed as dist
from tqdm import tqdm
import json
import pickle as pkl
import torch.multiprocessing as mp


def setup(rank, world_size, master_addr, master_port):
    print(f"Setting up rank: {rank}")
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Rank {rank} is setup")


def cleanup():
    dist.destroy_process_group()


def model_setup(rank, model_id, world_size):


    def cls_pooling(model_output, attention_mask):
        return model_output[0][:, 0]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    passage_model = AutoModel.from_pretrained(model_id).to(rank)

    return passage_model


def reindex(rank, machine_rank, world_size, master_addr, master_port, model_setup, model_id=None, overall_rank=None, filename="../datasets/nq/psgs_w100.tsv", file_size=21015324):
# def reindex(rank, machine_rank, world_size, master_addr, master_port, model_setup, model_id=None, overall_rank=None, filename="../datasets/nq/wikipedia_sarcasm_fact_distorted.tsv", file_size=971384):
    if world_size > 0:
        setup(overall_rank if overall_rank else machine_rank+rank, world_size, master_addr, master_port)
        print(rank, world_size, machine_rank+rank, filename)

    model = model_setup(rank, model_id, world_size)
    tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=512)
    model = model.to(rank)
    device = "cuda:"+str(rank)
    if world_size == 0:
        world_size += 1

    def read_tsv_lazy(filename, tokenizer, max_tokens, rank, world_size):
        print(filename)
        with open(filename, 'r') as file:
            next(file)  # skip first row
            batch = []
            max_len = 0
            for i, line in enumerate(file):
                if i % world_size != rank:
                    continue
                row = line.rstrip('\n').split('\t')
                try:
                    title, text, id = row[2], row[1], row[0]
                except Exception as e:
                    print(i, line)
                    print(e)
                    import sys
                    sys.exit()
                max_len = max(max_len, len(tokenizer("title: " + title + " passage: " + text[1:-1], truncation=True)["input_ids"]))
                if max_len * len(batch) >= max_tokens:
                    yield batch
                    batch = []
                    max_len = len(tokenizer("title: " + title + " passage: " + text[1:-1], truncation=True)["input_ids"])
                batch.append([title, text, int(id)])
            if batch:
                yield batch

    max_tokens = 135000


    data = []

    with torch.no_grad():
        with tqdm(total=file_size//world_size) as pbar:
            for i, batch in enumerate(read_tsv_lazy(filename, tokenizer, max_tokens, overall_rank if overall_rank else machine_rank+rank, world_size)):
                inputs = tokenizer(["title: " + title + " passage: " + text[1:-1] for title, text, _ in batch], return_tensors="pt", padding='longest', truncation=True)  # first and last character is always a quotation mark.
                inputs = {key: value.to(device) for key, value in inputs.items()}
                # inputs["input_ids"] = inputs.pop("input_ids")[:, :512].to(device)
                # inputs.update({"apply_mask": model.module.config.apply_question_mask, "extract_cls": model.module.config.extract_cls})
                text_features = model(**inputs)[0][:, 0].detach().cpu().numpy()
                [data.append([id, title + ": " + text[1:], text_features[i]]) for i, (title, text, id) in enumerate(batch)]
                pbar.update(len(batch))

    os.makedirs("wikipedia_embeddings", exist_ok=True)
    pkl.dump(data, open(f"wikipedia_embeddings/wikipedia_embeddings_bgem3_{overall_rank if overall_rank else machine_rank+rank}.pkl", "wb"))

# facts_distorted_sarcastic_
def run_index(world_size, master_addr, master_port, machine_index, model_setup, model_id=None, filename='../datasets/nq/psgs_w100.tsv', file_size=21015324):
# def run_index(world_size, master_addr, master_port, machine_index, model_setup, model_id=None, filename='../datasets/nq/wikipedia_sarcasm_fact_distorted.tsv', file_size=971384):
    world_size = world_size  # number of machines
    nprocs = torch.cuda.device_count()
    # model_id = "BAAI/llm-embedder"
    model_id = "BAAI/bge-m3"
    # model_id = "GPL/nq-distilbert-tas-b-gpl-self_miner"
    print(nprocs)
    mp.spawn(reindex,
             args=(nprocs*machine_index, world_size*nprocs, master_addr, master_port, model_setup, model_id, None, filename, file_size),
             nprocs=nprocs,
             join=True)

    # reindex(0, nprocs*machine_index, world_size*nprocs, master_addr, master_port, model_setup, model_id, None, filename, file_size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, required=True)
    parser.add_argument('--master_addr', type=str, required=True)
    parser.add_argument('--master_port', type=int, required=True)
    parser.add_argument('--machine_index', type=int, required=True)
    args = parser.parse_args()
    run_index(args.world_size, args.master_addr, args.master_port, args.machine_index, model_setup)
