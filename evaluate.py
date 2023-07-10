import itertools
import os
from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from mpire import WorkerPool
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel

from datasets import Dataset, load_dataset
from models.model import BabelNetTransformer
from utils.functions import (
    get_embeddings,
    get_vocab_index,
    cos,
    compute_retrieval_metric,
)


@hydra.main(config_path="configs", config_name="evaluate")
def main(cfg: DictConfig):
    print(f"Starting evaluation for task {cfg.task.name} for model {cfg.ckpt}")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = (
        BabelNetTransformer.load_from_checkpoint(cfg.ckpt).encoder
        if cfg.ckpt.endswith(".ckpt")
        else AutoModel.from_pretrained(cfg.ckpt)
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model.eval()
    if cfg.gpu != -1:
        model = model.to(f"cuda:{cfg.gpu}")

    num_layers = model.encoder.config.num_hidden_layers + 1
    layer_list = list(range(num_layers)) if cfg.layers == "all" else cfg.layers
    if isinstance(layer_list, int):
        layer_list = [cfg.layers]

    task_start = timer()
    if cfg.task.name == "tatoeba":
        lang_pair_to_score = eval_tatoeba(cfg, layer_list, model, tokenizer)

    if cfg.task.name == "bli":
        lang_pair_to_score = eval_bli(cfg, layer_list, model, tokenizer)

    if cfg.task.name == "xlsim":
        lang_pair_to_score = eval_xlsim(cfg, layer_list, model, tokenizer)

    task_end = timer()
    print(
        f"Time elapsed for {cfg.task.name} evaluation {timedelta(seconds=task_end - task_start)}"
    )

    results = pd.DataFrame(lang_pair_to_score, index=layer_list)
    results.index.name = "layer"

    output_dir = Path(to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {output_dir / f'res_{cfg.task.name}.csv'}")
    results.to_csv(output_dir / f"res_{cfg.task.name}.csv")


def eval_xlsim(cfg, layer_list, model, tokenizer):
    lang_pairs = sorted(
        list(
            set(
                [
                    x.split(".")[0].lower()
                    for x in os.listdir(to_absolute_path(cfg.task.path))
                ]
            )
        )
    )
    lang_pair_to_score = {}

    id_to_layer = {idx: layer for idx, layer in enumerate(layer_list)}
    for lang_pair in lang_pairs:
        print(f"Processing {lang_pair}")
        lang1, lang2 = [x.upper() for x in lang_pair.split("-")]
        lang_pair_to_score[lang_pair] = []

        src_dataset = Dataset.from_pandas(
            pd.read_csv(
                to_absolute_path(f"{cfg.task.path}/{lang_pair.upper()}.csv"),
                usecols=[lang1],
                header=0,
                keep_default_na=False,
            )
        ).map(lambda example: tokenizer(example[lang1]), desc="Tokenization")
        tgt_dataset = Dataset.from_pandas(
            pd.read_csv(
                to_absolute_path(f"{cfg.task.path}/{lang_pair.upper()}.csv"),
                usecols=[lang2],
                header=0,
                keep_default_na=False,
            )
        ).map(lambda example: tokenizer(example[lang2]), desc="Tokenization")

        scores = np.array(
            pd.read_csv(to_absolute_path(f"{cfg.task.path}/{lang_pair.upper()}.csv"))[
                "score"
            ]
        )

        src_embeddings = [
            x.cpu().numpy()
            for x in get_embeddings(
                model, src_dataset, tokenizer, cfg.gpu, layer_list, batch_size=256
            )
        ]
        tgt_embeddings = [
            x.cpu().numpy()
            for x in get_embeddings(
                model, tgt_dataset, tokenizer, cfg.gpu, layer_list, batch_size=256
            )
        ]

        for layer_id, layer in id_to_layer.items():
            cos_sim = cos(src_embeddings[layer_id], tgt_embeddings[layer_id])
            spearman_r = spearmanr(cos_sim, scores, nan_policy="omit")[0]
            print(
                f"Spearman's ρ score for {lang_pair} Layer {id_to_layer[layer_id]} : {spearman_r}"
            )
            lang_pair_to_score[lang_pair].append(spearman_r)
    return lang_pair_to_score


def eval_bli(cfg, layer_list, model, tokenizer):
    if cfg.task.bli_dataset == "all":
        dataset_to_folders = {
            dataset: os.listdir(to_absolute_path(path))
            for dataset, path in cfg.task.dataset_to_path.items()
        }
    else:
        dataset_to_folders = {
            cfg.task.bli_dataset: os.listdir(
                to_absolute_path(cfg.task.dataset_to_path[cfg.task.bli_dataset])
            )
        }
    print(f"Using datasets {list(dataset_to_folders.keys())}")

    lang_pair_to_score = {}
    for dataset, lang_pairs in dataset_to_folders.items():
        print(f"Processing dataset {dataset}")
        tgt_langs = sorted(list(set([p.split("-")[1] for p in lang_pairs])))
        langs = sorted(list(set(itertools.chain(*[x.split("-") for x in lang_pairs]))))
        lang_to_vocab = {
            lang: load_dataset(
                "text",
                data_files=to_absolute_path(f"{cfg.task.vocab_path}/{lang}_vocab.txt"),
                split="train",
            ).rename_column("text", "word")
            for lang in langs
        }
        print(f"Target languages {tgt_langs}")

        for tgt_lang in tgt_langs:
            sel_lang_pairs = [p for p in lang_pairs if p.split("-")[1] == tgt_lang]
            vocab_path = to_absolute_path(f"{cfg.task.vocab_path}/{tgt_lang}_vocab.txt")
            print(
                f"Target language {tgt_lang}, language pairs {sel_lang_pairs}, vocabulary file {vocab_path}"
            )

            tgt_lang_vocab_data = (
                lang_to_vocab[tgt_lang]
                .map(lambda example: tokenizer(example["word"], return_length=True))
                .filter(
                    lambda example: example["length"][0] >= 3
                    and example["input_ids"].count(tokenizer.sep_token_id) == 1,
                    desc="Filtering out 0-length words",
                )
            )
            print(f"Vocabulary length {len(tgt_lang_vocab_data)}")
            print(f"Computing {tgt_lang} vocabulary embedding")

            tgt_vocab_embeddings = [
                x.cpu().numpy()
                for x in get_embeddings(
                    model, tgt_lang_vocab_data, tokenizer, cfg.gpu, layer_list
                )
            ]
            tgt_vocab = list(tgt_lang_vocab_data["word"])

            for idx, lang_pair in enumerate(sel_lang_pairs):
                test_path = list(
                    Path(
                        to_absolute_path(
                            f"{cfg.task.dataset_to_path[dataset]}/{lang_pair}"
                        )
                    ).glob("*test*")
                )[0]

                test_sel_pair_df = pd.read_csv(
                    test_path, sep="\t", names=["word1", "word2"], keep_default_na=False
                )
                or_size = len(test_sel_pair_df)

                src_vocab = list(lang_to_vocab[lang_pair.split("-")[0]]["word"])
                test_sel_pair_df = (
                    test_sel_pair_df.query(
                        f"word2 in {tgt_vocab} or word2.str.lower() in {tgt_vocab}"
                    )
                    .query(f"word1 in {src_vocab} or word1.str.lower() in {src_vocab}")
                    .reset_index(drop=True)
                )
                print(
                    f"Discarded words for {dataset}_{lang_pair} = {or_size - len(test_sel_pair_df)}"
                )
                test_sel_pair_df["vocab_idx"] = test_sel_pair_df.word2.apply(
                    lambda x: get_vocab_index(x, tgt_vocab), tgt_vocab
                )
                test_src_word_dataset = Dataset.from_pandas(
                    pd.DataFrame(test_sel_pair_df["word1"])
                ).map(
                    lambda example: tokenizer(example["word1"]),
                    desc="Tokenization of test source words",
                )

                print(f"Computing test embeddings from {test_path}...")
                test_src_word_embeddings = [
                    x.cpu().numpy()
                    for x in get_embeddings(
                        model, test_src_word_dataset, tokenizer, cfg.gpu, layer_list
                    )
                ]

                lang_pair_to_score[f"{lang_pair}_{dataset}_vanilla"] = []
                tgt_vocab_idx = np.array(test_sel_pair_df["vocab_idx"])

                print(f"Starting retrieval...")
                start = timer()

                bli_input = [
                    [
                        test_src_word_embeddings[layer_id],
                        tgt_vocab_embeddings[layer_id],
                        tgt_vocab_idx,
                        "mrr",
                    ]
                    for layer_id, layer in enumerate(layer_list)
                ]

                print(f"Len of bli_input {len(bli_input)}")
                with WorkerPool(n_jobs=cfg.num_proc) as pool:
                    mrr_vanilla = pool.map(
                        compute_retrieval_metric, bli_input, progress_bar=True
                    )
                for layer_id, layer in enumerate(layer_list):
                    print(f"{lang_pair} - Layer {layer}: MRR {mrr_vanilla[layer_id]}")

                lang_pair_to_score[f"{lang_pair}_{dataset}_vanilla"] = mrr_vanilla
    return lang_pair_to_score


def eval_tatoeba(cfg, layer_list, model, tokenizer):
    lang_pairs = sorted(
        list(
            set([x.split(".")[1] for x in os.listdir(to_absolute_path(cfg.task.path))])
        )
    )
    id_to_layer = {idx: layer for idx, layer in enumerate(layer_list)}

    lang_pair_to_score = {}
    for lang_pair in lang_pairs:
        lang_pair_to_score[lang_pair] = []
        tgt_lang = lang_pair.split("-")[0]

        src_lines = [
            x.strip()
            for x in open(
                to_absolute_path(f"{cfg.task.path}/tatoeba.{lang_pair}.eng")
            ).readlines()
        ]
        tgt_lines = [
            x.strip()
            for x in open(
                to_absolute_path(f"{cfg.task.path}/tatoeba.{lang_pair}.{tgt_lang}")
            ).readlines()
        ]
        src_dataset = Dataset.from_dict({"src": src_lines}).map(
            lambda example: tokenizer(example["src"]), desc="Tokenization"
        )
        tgt_dataset = Dataset.from_dict({"tgt": tgt_lines}).map(
            lambda example: tokenizer(example["tgt"]), desc="Tokenization"
        )

        # Get src and tgt embeddings from each specified layer
        src_embeddings = [
            x.cpu().numpy()
            for x in get_embeddings(
                model, src_dataset, tokenizer, cfg.gpu, layer_list, batch_size=64
            )
        ]
        tgt_embeddings = [
            x.cpu().numpy()
            for x in get_embeddings(
                model, tgt_dataset, tokenizer, cfg.gpu, layer_list, batch_size=64
            )
        ]

        tatoeba_input = list(
            zip(src_embeddings, tgt_embeddings)
        )  # for every tuple in tatoeba_input, add a tensor made with
        # arange(0, len(tatoeba_input[0][0]))
        tgt_idx = np.arange(0, len(tgt_embeddings[0]))

        with WorkerPool(n_jobs=cfg.num_proc) as pool:
            results = pool.map(
                compute_retrieval_metric,
                [
                    (src_embeddings[idx], tgt_embeddings[idx], tgt_idx, "p1")
                    for idx, layer in id_to_layer.items()
                ],
                progress_bar=True,
            )

        for idx, score in enumerate(results):
            print(f"Accuracy score for {lang_pair} Layer {id_to_layer[idx]} : {score}")
            lang_pair_to_score[lang_pair].append(score)
    return lang_pair_to_score


if __name__ == "__main__":
    main()
