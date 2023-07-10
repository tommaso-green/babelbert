import csv
from collections import Counter
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import load_dataset, Dataset
from hydra.utils import to_absolute_path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.functions import get_sampler, update_language_counters


class BabelNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        train_path,
        alpha=0.5,
        val_train_overlap=False,
        random_seed=42,
        val_bli_file=None,
        vocab_dir=None,
        sel_langs="all",
        input_type="word_iso",
    ):
        super().__init__()
        self.val_train_overlap = val_train_overlap
        self.tgt_vocab_data = {}
        self.tgt_word_vocab_idx = {}
        self.model_name = model_name

        self.train_path = train_path

        self.val_bli_file = val_bli_file
        self.vocab_dir = to_absolute_path(vocab_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.batch_size = batch_size
        self.alpha = alpha
        self.sampler = None

        self.full_dataset = None
        self.train_dataset = None
        self.val_bli_dataset = None

        self.random_seed = random_seed

        self.sel_langs = sel_langs
        self.lang_pair_counter = Counter()
        self.lang_counter = Counter()
        self.distinct_words = Counter()

        self.data_is_ready = False
        self.tgt_words = {}
        self.input_type = input_type

        self.lang_to_id = None
        self.id_to_lang = None

    def prepare_data(self):
        if not self.data_is_ready:  # If function is called for the first time
            # Prepare training set
            self.full_dataset = load_dataset(
                "csv", data_files={"train": self.train_path}, keep_default_na=False
            )
            if self.sel_langs != "all":
                print(f"Filtering for languages {self.sel_langs}")
                self.full_dataset = self.full_dataset.filter(
                    lambda example: example["lang1"] in self.sel_langs
                    and example["lang2"] in self.sel_langs,
                    desc=f"filtering for languages {self.sel_langs}",
                )
            print(f"Total number of synonym pairs {len(self.full_dataset['train'])}")

            # Prepare validation set with bli data
            tgt_vocab = {}
            val_words = []
            for lang_pair, bli_filepath in self.val_bli_file.items():
                tgt_lang = lang_pair.split("-")[1]
                print(f"Processing {lang_pair} BLI data, target language {tgt_lang}")
                vocab_path = to_absolute_path(f"{self.vocab_dir}/{tgt_lang}_vocab.txt")
                self.tgt_vocab_data[tgt_lang] = load_dataset(
                    "text", data_files=vocab_path, split="train"
                ).rename_column("text", "word")
                self.tgt_vocab_data[tgt_lang] = self.tgt_vocab_data[tgt_lang].map(
                    lambda example: self.tokenizer(example["word"], return_length=True),
                    desc=f"Tokenization of {tgt_lang} vocabulary",
                    batched=True,
                )
                self.tgt_vocab_data[tgt_lang] = self.tgt_vocab_data[tgt_lang].filter(
                    lambda example: example["length"][0] >= 3
                    and example["input_ids"].count(self.tokenizer.sep_token_id) == 1,
                    desc="Filtering out 0-length words",
                )
                tgt_vocab[tgt_lang] = list(self.tgt_vocab_data[tgt_lang]["word"])
                self.tgt_words[tgt_lang] = self.tgt_vocab_data[tgt_lang]["word"]
                val_bli_df = pd.read_csv(
                    to_absolute_path(bli_filepath),
                    sep="\t",
                    names=["word1", "word2"],
                    keep_default_na=False,
                )
                val_words += val_bli_df["word1"].tolist() + val_bli_df["word2"].tolist()
                or_size = len(val_bli_df)
                val_bli_df = val_bli_df.query(
                    f"word2 in {tgt_vocab[tgt_lang]} or word2.str.lower() in {tgt_vocab[tgt_lang]}"
                ).reset_index(drop=True)
                print(
                    f"*** {or_size - len(val_bli_df)} words were deleted (not present in target language vocabulary)"
                )
                self.tgt_word_vocab_idx[tgt_lang] = np.array(
                    [
                        tgt_vocab[tgt_lang].index(x)
                        if x in tgt_vocab[tgt_lang]
                        else tgt_vocab[tgt_lang].index(x.lower())
                        for x in list(val_bli_df["word2"])
                    ]
                )
                self.full_dataset[f"val_bli_{lang_pair}"] = Dataset.from_pandas(
                    pd.DataFrame(val_bli_df["word1"])
                ).map(
                    lambda example: self.tokenizer(example["word1"]),
                    desc=f"Tokenization of bli val words (lang pair {lang_pair})",
                    batched=True,
                )

            # If val_train_overlap is False, delete validation words from training set
            if not self.val_train_overlap:
                val_words = set(val_words)
                old_train_size = len(self.full_dataset["train"])
                self.full_dataset["train"] = self.full_dataset["train"].filter(
                    lambda example: {example["word1"], example["word2"]}.isdisjoint(
                        val_words
                    ),
                    desc="Deleting val words from training set",
                )
                new_train_size = len(self.full_dataset["train"])
                print(f"> Old training set size {old_train_size}")
                print(f"> New training set size {new_train_size}")
                print(f"> Deleted pairs {old_train_size - new_train_size}")
            else:
                print("+ Validation and training sets may overlap +")

            # Preprocess training set
            self.full_dataset["train"] = self.full_dataset["train"].map(
                self.preprocess, desc="Tokenization of training set"
            )
            if self.input_type == "word_iso":
                self.full_dataset["train"] = self.full_dataset["train"].filter(
                    lambda example: all(l >= 3 for l in example["length"]),
                    desc="Filtering out 0-length words",
                )

            self.data_is_ready = True

    def preprocess(self, example):
        if self.input_type == "word_iso":
            # Word fed in isolation, [CLS] w_1 [SEP]
            enc = self.tokenizer(
                [example["word1"], example["word2"]], return_length=True
            )
        if self.input_type == "w1-s1-w2-s2":
            # Everything fed in the same forward pass, i.e. [CLS] w_1 [SEP] s_1 [SEP] w_2 [SEP] s_2 [SEP]
            enc = self.tokenizer(
                example["word1"]
                + self.tokenizer.sep_token
                + example["word2"]
                + self.tokenizer.sep_token
                + example["sent1"]
                + self.tokenizer.sep_token
                + example["sent2"],
                return_length=True,
                truncation=True,
            )
            mask = torch.zeros(
                torch.tensor(enc["input_ids"]).shape,
                dtype=torch.tensor(enc["input_ids"]).dtype,
            )
            idx = torch.where(
                torch.tensor(enc["input_ids"]) == self.tokenizer.sep_token_id
            )[0]
            mask[1 : idx[0]] = 1
            mask[idx[0] + 1 : idx[1]] = 2
            enc["pooling_mask"] = mask.tolist()
        if self.input_type == "w1-s1|w2-s2":
            # Word and sentence fed in the same forward pass, but pair 1 and 2 are in different forward passes
            # i.e. input_1 = [CLS] w_1 [SEP] s_1 [SEP], input_2 = [CLS] w_2 [SEP] s_2 [SEP]
            # Only use word representations, i.e. mean pooling only on the subwords of w_1 and w_3
            enc = self.tokenizer(
                [example["word1"], example["word2"]],
                [example["sent1"], example["sent2"]],
                return_length=True,
                truncation=True,
            )
            mask = torch.zeros([2, max(enc["length"])], dtype=torch.int64)
            idx_w1 = torch.where(
                torch.tensor(enc["input_ids"][0]) == self.tokenizer.sep_token_id
            )[0][0]
            idx_w2 = torch.where(
                torch.tensor(enc["input_ids"][1]) == self.tokenizer.sep_token_id
            )[0][0]
            mask[0, 1:idx_w1] = 1
            mask[1, 1:idx_w2] = 1
            enc["pooling_mask"] = mask.tolist()
        if self.input_type == "w1-s1|w2-s2_mt":
            # As above, Word and sentence fed in the same forward pass, but pair 1 and 2 are in different forward
            # passes i.e. input_1 = [CLS] w_1 [SEP] s_1 [SEP], input_2 = [CLS] w_2 [SEP] s_2 [SEP] Use both word and
            # sentence representations, in a multi-task setting (contrastive loss on words and sentences)
            enc = self.tokenizer(
                [example["word1"], example["word2"]],
                [example["sent1"], example["sent2"]],
                return_length=True,
                truncation=True,
                return_special_tokens_mask=True,
            )
            mask = [
                torch.logical_not(torch.tensor(x)).long()
                for x in enc["special_tokens_mask"]
            ]
            sent_start_idx_1 = (
                torch.where(
                    torch.tensor(enc["input_ids"][0]) == self.tokenizer.sep_token_id
                )[0][-2].item()
                + 1
            )
            sent_start_idx_2 = (
                torch.where(
                    torch.tensor(enc["input_ids"][1]) == self.tokenizer.sep_token_id
                )[0][-2].item()
                + 1
            )
            mask[0][sent_start_idx_1:-1] = 2
            mask[1][sent_start_idx_2:-1] = 2
            enc["pooling_mask"] = torch.nn.utils.rnn.pad_sequence(
                mask, batch_first=True, padding_value=0
            ).tolist()

        return enc

    def setup(self, stage=None):
        print(f"Running datamodule setup for stage {stage}")
        if stage == "fit":
            train_label_to_synset = {
                label: synset
                for label, synset in enumerate(
                    self.full_dataset["train"].unique("synsetID")
                )
            }
            synset_to_train_label = {
                synset: label for label, synset in train_label_to_synset.items()
            }
            self.full_dataset["train"] = self.full_dataset["train"].map(
                lambda example: {"label": synset_to_train_label[example["synsetID"]]},
                desc="Mapping synset ids to numeric labels. (train split)",
            )
            langs = sorted(
                list(
                    set(self.full_dataset["train"].unique("lang1"))
                    | set(self.full_dataset["train"].unique("lang2"))
                )
            )
            self.lang_to_id = dict(zip(langs, range(len(langs))))
            self.id_to_lang = {value: key for key, value in self.lang_to_id.items()}
            self.full_dataset["train"] = self.full_dataset["train"].map(
                lambda example: {
                    "lang1_code": self.lang_to_id[example["lang1"]],
                    "lang2_code": self.lang_to_id[example["lang2"]],
                }
            )

            self.full_dataset.set_format()
            if self.alpha == -1:
                self.sampler = None
                print("No upsampling (alpha=-1)")
            else:
                print("Setting upsampling strategy")
                self.sampler = get_sampler(self.full_dataset["train"], self.alpha)
            columns = [
                "input_ids",
                "attention_mask",
                "label",
                "lang1_code",
                "lang2_code",
            ]
            if "pooling_mask" in self.full_dataset.column_names["train"]:
                columns.append("pooling_mask")
            self.full_dataset["train"].set_format(type="torch", columns=columns)
            self.train_dataset = self.full_dataset["train"]

        for col in [split for split in self.full_dataset.keys() if "bli" in split]:
            self.full_dataset[col].set_format(
                type="torch", columns=["input_ids", "attention_mask"]
            )

    def compute_stats(self, split, path=None, write_to_file=False, languages=None):
        if languages is None:
            languages = sorted(
                list(
                    set(self.full_dataset[split].unique("lang1")).union(
                        set(self.full_dataset[split].unique("lang2"))
                    )
                )
            )
        language_pairs = set(combinations_with_replacement(set(languages), 2))

        column_list = ["languages"] + languages + ["total", "distinct_words"]

        self.lang_pair_counter = Counter({lang_pair: 0 for lang_pair in language_pairs})
        self.lang_counter = Counter({lang: 0 for lang in languages})
        self.distinct_words = {key: set() for key in languages}

        # Compute stats for language pairs
        self.full_dataset[split].map(
            update_language_counters,
            fn_kwargs={
                "lang_pair_counter": self.lang_pair_counter,
                "lang_counter": self.lang_counter,
                "distinct_words": self.distinct_words,
            },
            desc="Counting samples per language pair for stats",
        )

        # Write csv file for the selected split
        if write_to_file:
            rows_to_write = [column_list]
            for lang1 in languages:
                lang_1_values = [lang1]
                tot_pairs_lang1 = 0
                for lang2 in languages:
                    pair = (
                        (lang1, lang2)
                        if (lang1, lang2) in self.lang_pair_counter
                        else (lang2, lang1)
                    )
                    value = self.lang_pair_counter[pair]
                    tot_pairs_lang1 += value
                    lang_1_values.append(value)
                lang_1_values += [tot_pairs_lang1, len(self.distinct_words[lang1])]
                rows_to_write.append(lang_1_values)

            with open(path + "/" + split + "_stats.csv", mode="w") as file:
                writer = csv.writer(
                    file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                writer.writerows(rows_to_write)
                file.close()

    def save_and_compute_stats(self, path):
        self.full_dataset.save_to_disk(path)
        for split in self.full_dataset.keys():
            self.compute_stats(split, path, write_to_file=True)

    def train_dataloader(self):
        if self.input_type in ["word_iso", "w1-s1|w2-s2", "w1-s1|w2-s2_mt"]:
            padding_func = self.pad_batch
        if self.input_type == "w1-s1-w2-s2":
            padding_func = self.pad_batch_single_line

        shuffle = True if self.sampler is None else False
        print(f"Shuffle set to: {shuffle}")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=padding_func,
            sampler=self.sampler,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        val_loaders = [
            DataLoader(
                self.full_dataset[f"val_bli_{lang_pair}"],
                batch_size=self.batch_size,
                collate_fn=self.pad_batch_single_line,
            )
            for lang_pair in self.val_bli_file.keys()
        ]
        val_loaders += [
            DataLoader(
                vocab_data,
                batch_size=self.batch_size,
                collate_fn=self.pad_batch_single_line,
            )
            for tgt_lang, vocab_data in self.tgt_vocab_data.items()
        ]

        self.loader_idx_to_name = {
            idx: "bli_" + lang_pair
            for idx, lang_pair in enumerate(self.val_bli_file.keys())
        }
        self.loader_idx_to_name.update(
            {
                idx + len(self.loader_idx_to_name): "vocab_" + tgt_lang
                for idx, tgt_lang in enumerate(self.tgt_vocab_data.keys())
            }
        )

        return val_loaders

    def pad_batch_single_line(self, batch):
        whole_batch = {
            "input_ids": [example["input_ids"] for example in batch],
            "attention_mask": [example["attention_mask"] for example in batch],
        }
        whole_batch = self.tokenizer.pad(whole_batch, return_tensors="pt")
        if "pooling_mask" in batch[0].keys():
            whole_batch["pooling_mask"] = pad_sequence(
                [example["pooling_mask"] for example in batch], batch_first=True
            )
        if "label" in batch[0].keys():
            whole_batch["label"] = torch.cat(
                [example["label"].repeat(2) for example in batch]
            )
        return whole_batch

    def pad_batch(self, batch):
        whole_batch = {
            "input_ids": [x for example in batch for x in example["input_ids"]],
            "attention_mask": [
                x for example in batch for x in example["attention_mask"]
            ],
        }
        whole_batch = self.tokenizer.pad(whole_batch, return_tensors="pt")
        if "label" in batch[0].keys():
            whole_batch["label"] = torch.cat(
                [example["label"].repeat(2) for example in batch]
            )
        if "pooling_mask" in batch[0].keys():
            whole_batch["pooling_mask"] = pad_sequence(
                [x for example in batch for x in example["pooling_mask"]],
                batch_first=True,
            )
        if "lang1_code" in batch[0].keys() and "lang2_code" in batch[0].keys():
            lang1_codes = [x["lang1_code"] for x in batch]
            lang2_codes = [x["lang2_code"] for x in batch]
            whole_batch["lang_code"] = torch.stack(
                [item for sublist in zip(lang1_codes, lang2_codes) for item in sublist]
            )

        return whole_batch
