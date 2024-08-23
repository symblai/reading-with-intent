from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from functools import partial
from train import sentiment_dataset, sarcasm_dataset, val_loop
import os
import torch.distributed as dist


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8085'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



if __name__ == '__main__':
    setup(0, 1)
    model_name = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).cuda()
    model.load_state_dict({key.split("module.")[1]: value for key, value in torch.load("sarc_roberta-base_classifier_epoch_9.pt", map_location="cuda").state_dict().items()})
    # model_name = "/" + model_name

    dataset_name = "sarc"
    sarc_val = sentiment_dataset(dataset_name, tokenizer, "validation")
    sarc_val_synth1 = sarcasm_dataset("sarcasm_val_dataset.json", tokenizer)
    sarc_val_synth2 = sarcasm_dataset("sarcasm_val_dataset_natural_retrieve.json", tokenizer)

    val_collator = partial(sentiment_dataset.collator_fn, max_size=512)
    sarc_dl_val = DataLoader(sarc_val, batch_size=100, shuffle=False, num_workers=4, drop_last=False, collate_fn=val_collator, pin_memory=True)
    sarc_dl_val_synth1 = DataLoader(sarc_val_synth1, batch_size=50, shuffle=False, num_workers=4, drop_last=False, collate_fn=val_collator, pin_memory=True)
    sarc_dl_val_synth2 = DataLoader(sarc_val_synth2, batch_size=50, shuffle=False, num_workers=4, drop_last=False, collate_fn=val_collator, pin_memory=True)


    model.eval()
    with torch.no_grad():
        # sarc_dl_val.dataset.evaluate_results(val_loop(0, model, sarc_dl_val))
        print("Natural Retrieve:")
        if not os.path.exists("results_on_nq_psa.pt"):
            results = val_loop(0, model, sarc_dl_val_synth2)
        else:
            results = torch.load("results_on_nq_psa.pt")
        import IPython; IPython.embed()
        print("Random Subset:")
        sarc_dl_val_synth1.dataset.evaluate_results(val_loop(0, model, sarc_dl_val_synth1))

