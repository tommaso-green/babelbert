from collections import Counter
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import torch
import tqdm
from pytorch_metric_learning.distances.cosine_similarity import CosineSimilarity
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_embeddings(model, dataset, tokenizer, gpu, layers, batch_size=64):
    dataloader = DataLoader(
        dataset,
        batch_size,
        collate_fn=partial(pad_input, tokenizer=tokenizer, single_element=True),
    )
    embeddings = [[] for _ in range(len(layers))]
    for batch in tqdm.tqdm(dataloader, desc="Getting embeddings"):
        with torch.no_grad():
            if gpu != -1:
                input_ids = batch["input_ids"].to(f"cuda:{gpu}")
                attention_mask = batch["attention_mask"].to(f"cuda:{gpu}")
            else:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        for layer_id, layer in enumerate(layers):
            hidden_states = output.hidden_states[layer]
            word_repr = mean_pooling(hidden_states, attention_mask.clone())
            embeddings[layer_id].append(word_repr)

    for layer_id, layer in enumerate(layers):
        embeddings[layer_id] = torch.cat(embeddings[layer_id])
    return embeddings


def get_vocab_index(x, tgt_vocab):
    try:
        idx = tgt_vocab.index(x)
    except:
        idx = tgt_vocab.index(x.lower())
    return idx


def retrieval_func(cos_sim, metric):
    pred = np.argsort(-cos_sim, axis=1)
    if metric == "mrr":
        score = (
            1 / (np.argwhere(pred == np.arange(pred.shape[0])[:, np.newaxis])[:, 1] + 1)
        ).mean()
    if metric == "p@1":
        score = (pred[:, 0] == np.arange(pred.shape[0])).mean()
    return score


def pad_input(batch, tokenizer, single_element=False):
    if single_element:
        whole_batch = {
            "input_ids": [example["input_ids"] for example in batch],
            "attention_mask": [example["attention_mask"] for example in batch],
        }
    else:
        whole_batch = {
            "input_ids": [x for example in batch for x in example["input_ids"]],
            "attention_mask": [
                x for example in batch for x in example["attention_mask"]
            ],
        }
    whole_batch = tokenizer.pad(whole_batch, return_tensors="pt")
    return whole_batch


def cos(x, y):
    return (x * y).sum(axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))


def mean_pooling(hidden_states, attention_mask, seq_len_axis=1):
    sep_pos = attention_mask.sum(axis=1) - 1
    attention_mask.scatter_(dim=1, index=sep_pos.unsqueeze(1), value=0)
    attention_mask[:, 0] = 0

    masked_states = hidden_states * attention_mask[:, :, None]
    word_repr_sum = masked_states.sum(dim=seq_len_axis)
    word_repr_mean = word_repr_sum / (sep_pos - 1)[:, None]

    return word_repr_mean


def cos_sim_nn(src_repr, tgt_repr):
    return np.argsort(cosine_similarity(src_repr, tgt_repr))[:, ::-1]


def compute_p1(pred, tgt_idx):
    return (pred[:, 0] == tgt_idx).mean()


def compute_mrr(pred, tgt_idx):
    mrr = (1 / (np.argwhere(pred == tgt_idx[:, None])[:, 1] + 1)).mean()
    return mrr


def compute_retrieval_metric(src_repr, tgt_repr, tgt_idx, metric_name):
    pred = cos_sim_nn(src_repr, tgt_repr)
    if metric_name == "mrr":
        score = compute_mrr(pred, tgt_idx)
    if metric_name == "p1":
        score = compute_p1(pred, tgt_idx)
    return score


class CSLSSimilarity(CosineSimilarity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.is_inverted
        assert self.normalize_embeddings

    def compute_mat(self, query_emb, ref_emb):
        cos_sim = super().compute_mat(query_emb, ref_emb)
        mask = ~torch.eye(cos_sim.shape[0], dtype=bool, device=cos_sim.device)
        masked_sim = cos_sim * mask
        mean_cos = masked_sim.sum(1) / mask.sum(1)
        return 2 * cos_sim - mean_cos[:, None] - mean_cos


def compute_weights(example, idx, lang_pair_to_sample_weight, samples_weight):
    language_pair = frozenset([example["lang1"], example["lang2"]])
    samples_weight[idx] = lang_pair_to_sample_weight[language_pair]


def get_sampler(dataset, alpha):
    # For every language pair compute its relative frequency p_i, the norm_factor and the new probabilities
    # q_i = p_i ^ alpha / norm_factor
    # Since class weights are conditional probabilities, the weights are 1/(len(dataset)*q_i)
    languages = set(dataset.unique("lang1")).union(set(dataset.unique("lang2")))
    language_pairs = [
        frozenset(x) for x in combinations_with_replacement(set(languages), 2)
    ]
    lang_pair_counter = Counter({lang_pair: 0 for lang_pair in language_pairs})
    dataset.map(
        count_language_pairs,
        fn_kwargs={"lang_pair_counter": lang_pair_counter},
        desc="Counting samples per language pair for stats",
    )

    total_num_pairs = sum(list(lang_pair_counter.values()))

    lang_pair_to_freq = {
        key: lang_pair_counter[key] / total_num_pairs
        for key in lang_pair_counter.keys()
    }
    norm_factor = np.sum([p**alpha for p in lang_pair_to_freq.values()])
    lang_pair_to_sample_weight = {
        lang_pair: (p**alpha / norm_factor) / lang_pair_counter[lang_pair]
        if p != 0
        else 0
        for lang_pair, p in lang_pair_to_freq.items()
    }

    # Assign to every synonym pair its weight, according to its language pair
    samples_weight = torch.empty(len(dataset))
    dataset.map(
        compute_weights,
        fn_kwargs={
            "lang_pair_to_sample_weight": lang_pair_to_sample_weight,
            "samples_weight": samples_weight,
        },
        with_indices=True,
        desc="Assigning sample weights",
    )
    # Create the sampler to be used in the dataloader
    return WeightedRandomSampler(samples_weight, len(samples_weight))


def update_language_counters(
    example, lang_pair_counter, lang_counter, distinct_words=None
):
    if (example["lang1"], example["lang2"]) in lang_pair_counter.keys():
        pair = (example["lang1"], example["lang2"])
    else:
        pair = (example["lang2"], example["lang1"])
    lang_pair_counter[pair] += 1
    lang_counter[pair[0]] += 1
    lang_counter[pair[1]] += 1


def count_language_pairs(example, lang_pair_counter):
    lang_pair_counter[frozenset([example["lang1"], example["lang2"]])] += 1
