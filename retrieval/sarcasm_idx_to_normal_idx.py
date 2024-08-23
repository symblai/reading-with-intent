import pickle as pkl
import csv
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd

retrieval_results = pkl.load(open("../retrieval/gpl_retrieval_results_w_passage_sarcastic_lies.pkl", "rb"))
rows = []
with open("../datasets/nq/wikipedia_sarcasm_fact_distorted_cleaned.tsv", "r", encoding='utf-8') as tsvfile:
    for i, line in enumerate(tsvfile):
        row = line.rstrip('\n').split('\t')
        title, text, id = row[2], row[1], row[0]
        rows.append([title, text, id])

rows = rows[1:]

all_passages = [retrieval_results[i]["ctxs"][j] for i in range(len(retrieval_results)) for j in range(len(retrieval_results[i]["ctxs"]))]
all_passages_txt = np.array([i["text"].replace("\n", " ") for i in all_passages])
df = pd.DataFrame(all_passages_txt, columns=['text'])
# all_passages_txt = np.array([i.replace("\t", " ") for i in all_passages_txt])

def find_equiv_row(row, index):
    transformations = [
        lambda x: x,
        lambda x: x.replace("\n", " "),
        lambda x: x.replace("\n", " ").replace('""', '"').strip("\""),
        lambda x: x.replace("\n", " ").replace('""', '"'),
        lambda x: x.replace("\n", " ").strip("\""),
        lambda x: x.replace("\n", " ").replace('""', '"').lstrip("\""),
        lambda x: x.replace("\n", " ").replace('""', '"').rstrip("\""),
    ]
    # Apply transformations sequentially
    for transform in transformations:
        transformed_text = transform(row[1])
        equiv_row = np.argwhere(transformed_text == all_passages_txt)
        if equiv_row.shape[0]:
            return equiv_row
    try:
        # if index == 4293 or index == 5723:
        #     import ipdb; ipdb.set_trace()
        indiv_words = row[1].split(" ")
        bad_pos = np.argwhere(["." in i or "\"" in i or "'" in i for i in indiv_words])
        if bad_pos.shape[0] == 0:
            starting_pos = [5]
            ending_pos = [min(35, len(indiv_words))]
        elif bad_pos.shape[0] == 1:
            if bad_pos[0][0] + 30 > len(indiv_words):
                starting_pos = [5]
                ending_pos = [max(20, bad_pos[0][0])]
            else:
                starting_pos = [max(bad_pos[0][0] + 2, 2)]
                ending_pos = [starting_pos[0] + 30]
        else:
            len_bad_poses = bad_pos[1:] - bad_pos[:-1]
            init_bad_pos = np.argmax(len_bad_poses)
            starting_pos = [max(bad_pos[init_bad_pos][0] + 2, 2)]
            ending_pos = [max(bad_pos[init_bad_pos+1][0] - 2, starting_pos[0] + 10) if len(bad_pos) > 1 else (starting_pos[0] + 10)]
            if ending_pos[0] - starting_pos[0] < 40 and len(bad_pos) > 2:
                init_bad_pos = np.argsort((len_bad_poses).reshape(-1))[-2]
                starting_pos.append(max(bad_pos[init_bad_pos][0] + 2, 2))
                ending_pos.append(max(bad_pos[init_bad_pos+1][0] - 2, starting_pos[1] + 10) if len(bad_pos) > 1 else (starting_pos[1] + 10))
    except Exception:
        print(index)
        raise Exception
    search_idx = []
    for i in range(len(starting_pos)):
        search_term = " ".join(indiv_words[starting_pos[i]:ending_pos[i]])
        search_idx.append(df[df['text'].str.contains(search_term, regex=False)].index.to_numpy())
    if len(search_idx) == 1:
        return search_idx[0]
    else:
        return np.intersect1d(*search_idx)


def wrapper(params):
    index, row = params
    return find_equiv_row(row, index)

# Use multiprocessing to parallelize the computation
if __name__ == "__main__":
    print(cpu_count())
    params = [(index, row) for index, row in enumerate(rows)]
    with Pool(cpu_count()-10) as p:
        equiv_rows = list(tqdm(p.imap(wrapper, params), total=len(rows)))

    # equiv_rows = [find_equiv_row(rows[i], i) for i in tqdm(range(len(rows[299000:301000])))]
    # equiv_rows = [find_equiv_row(rows[i], i) for i in tqdm(range(4292, 5724))]

    import IPython; IPython.embed()
# Now equiv_row contains the equivalent rows found using multiprocessing
