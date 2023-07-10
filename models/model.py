import collections
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import pytorch_lightning as pl
import torch
import transformers
import wandb
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import SupConLoss
from sklearn.metrics.pairwise import cosine_similarity
from torch_scatter import scatter_mean
from torchmetrics import RetrievalMRR, RetrievalPrecision
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from utils.functions import CSLSSimilarity


class BabelNetTransformer(pl.LightningModule):
    def __init__(
        self,
        model_name,
        training_type,
        learning_rate,
        loss="ntxent",
        word_repr_type="mean_pool",
        temperature=0.07,
        reduction_factor=None,
        warmup_peak=-1,
        max_epochs=0,
        checkpoint_file="",
        similarity="cos",
        layerwise_averaging=False,
        weight_decay=0.0,
    ):
        super().__init__()

        self.training_type = training_type

        if self.training_type == "adapters":  # Adapter fine-tuning
            self.encoder = AutoModel.from_pretrained(
                checkpoint_file if checkpoint_file is not None else model_name
            )

            adapter_config = transformers.PfeifferConfig(
                reduction_factor=reduction_factor
            )

            # add a new adapter
            self.encoder.add_adapter("mss", config=adapter_config)

            # Enable adapter training
            self.encoder.train_adapter("mss")

        if self.training_type == "fft":  # Full fine-tuning
            self.encoder = AutoModel.from_pretrained(
                checkpoint_file if checkpoint_file is not None else model_name
            )

        self.emb_size = self.encoder.config.hidden_size
        self.learning_rate = learning_rate
        self.warmup_peak = warmup_peak
        self.weight_decay = weight_decay
        self.layerwise_averaging = layerwise_averaging

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sep_token_id = tokenizer.sep_token_id
        self.register_buffer(
            "special_token_ids",
            torch.tensor(
                [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
            ),
        )
        if similarity == "cos":
            self.similarity = CosineSimilarity()
        if similarity == "csls":
            self.similarity = CSLSSimilarity()
        if loss == "ntxent":
            self.loss_func = losses.NTXentLoss(
                temperature=temperature, distance=self.similarity
            )
        if loss == "supcon":
            self.loss_func = SupConLoss(
                temperature=temperature, distance=self.similarity
            )

        self.max_epochs = max_epochs

        self.val_prec_at_1 = RetrievalPrecision(k=1)
        self.val_mrr = RetrievalMRR()
        self.vanilla_bli_scores = {}
        self.validation_checks = 0

        self.lang_counter = collections.Counter()

        self.word_repr_type = word_repr_type  # whether to use [CLS] or mean pooling
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        output = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=self.layerwise_averaging,
        )
        hidden_states = (
            torch.stack(output.hidden_states)
            if self.layerwise_averaging
            else output.last_hidden_state
        )

        repr = mean_pooling(
            hidden_states,
            batch,
            self.trainer.datamodule.input_type,
            self.layerwise_averaging,
        )

        if self.trainer.datamodule.input_type == "w1-s1|w2-s2_mt":
            word_repr = repr[:, 1, :]
            sent_repr = repr[:, 2, :]
        else:
            word_repr = repr

        loss = self.loss_func(word_repr, batch["label"])
        if self.trainer.datamodule.input_type == "w1-s1|w2-s2_mt":
            loss += self.loss_func(sent_repr, batch["label"])
        cos_sim = cosine_similarity(
            word_repr.cpu().detach().numpy(), word_repr.cpu().detach().numpy()
        )
        np.fill_diagonal(cos_sim, 0)

        self.lang_counter.update(
            [self.trainer.datamodule.id_to_lang[x] for x in batch["lang_code"].tolist()]
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log(
            "train/avg_cos_sim",
            (cos_sim.sum(axis=1) / (cos_sim.shape[0] - 1)).mean(),
            on_step=True,
            on_epoch=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        output = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=self.layerwise_averaging,
        )

        hidden_states = (
            torch.stack(output.hidden_states)
            if self.layerwise_averaging
            else output.last_hidden_state
        )
        word_repr = mean_pooling(
            hidden_states, batch, "word_iso", self.layerwise_averaging
        )
        return {"word_repr": word_repr}

    def validation_epoch_end(self, outputs):
        start = timer()
        bli_diff = []

        for lang_pair in self.trainer.datamodule.val_bli_file.keys():
            tgt_lang = lang_pair.split("-")[1]
            print(f"Evaluating {lang_pair} BLI, target language {tgt_lang}")
            bli_loader_idx = [
                idx
                for idx, name in self.trainer.datamodule.loader_idx_to_name.items()
                if "bli" in name and lang_pair in name
            ][0]
            tgt_loader_idx = [
                idx
                for idx, name in self.trainer.datamodule.loader_idx_to_name.items()
                if "vocab" in name and tgt_lang in name
            ][0]
            src_word_repr = (
                torch.cat([x["word_repr"] for x in outputs[bli_loader_idx]])
                .cpu()
                .numpy()
            )
            tgt_word_repr = (
                torch.cat([x["word_repr"] for x in outputs[tgt_loader_idx]])
                .cpu()
                .numpy()
            )

            cos_sim = cosine_similarity(src_word_repr, tgt_word_repr)
            pred = np.argsort(-cos_sim, axis=1)
            bli_score = (
                1
                / (
                    np.argwhere(
                        pred
                        == self.trainer.datamodule.tgt_word_vocab_idx[tgt_lang][:, None]
                    )[:, 1]
                    + 1
                )
            ).mean()
            sorted_sim = np.sort(cos_sim)[:, ::-1][:, :100]
            print(f">>> MRR score for BLI {lang_pair} : {bli_score}")
            if self.trainer.global_step == 0 and isinstance(
                self.trainer.logger, pl.loggers.wandb.WandbLogger
            ):
                wandb.define_metric(f"val/mrr_bli_tr_{lang_pair}", summary="max")
                wandb.define_metric(f"val/avg_cos_sim_tr_{lang_pair}", summary="mean")

            if self.trainer.global_step == 0:
                self.vanilla_bli_scores[lang_pair] = bli_score

            diff = (
                bli_score - self.vanilla_bli_scores[lang_pair]
            ) / self.vanilla_bli_scores[lang_pair]
            bli_diff.append(diff)
            self.log(f"val/mrr_bli_tr_{lang_pair}", bli_score)
            self.log(f"val/avg_cos_sim_tr_{lang_pair}", sorted_sim.mean())
        end = timer()

        avg_bli_diff = sum(bli_diff) / len(bli_diff)
        print(
            f"+++ Arithmetic mean of  BLI diff of {len(bli_diff)} language pairs: {avg_bli_diff}"
        )
        print(f"Time elapsed for bli evaluation {timedelta(seconds=end - start)}")

        if self.trainer.global_step == 0 and isinstance(
            self.trainer.logger, pl.loggers.wandb.WandbLogger
        ):
            wandb.define_metric(f"val/mrr_bli_tr_avg_diff", summary="max")

        self.log(f"val/mrr_bli_tr_avg_diff", avg_bli_diff)

        if self.validation_checks >= 1:
            lang_freq_data = self.lang_counter.most_common()
            self.trainer.logger.log_table(
                key=f"train/seen_langs",
                data=lang_freq_data,
                columns=["lang", "num_examples"],
            )
        self.validation_checks += 1
        print(f"Done")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.warmup_peak != -1:
            train_steps = self.trainer.estimated_stepping_batches
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(train_steps * self.warmup_peak),
                num_training_steps=train_steps * self.max_epochs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            return optimizer


def mean_pooling(hidden_states, batch, input_type, layerwise_averaging):
    if input_type in "word_iso":
        pooling_mask = batch["attention_mask"]
        sep_pos = pooling_mask.sum(axis=1) - 1
        subword_mask = torch.scatter(
            pooling_mask, dim=1, index=sep_pos.unsqueeze(1), value=0
        )
        subword_mask[:, 0] = 0

        masked_states = hidden_states * subword_mask[:, :, None]
        if layerwise_averaging:
            word_repr_sum = masked_states.sum(dim=2)
        else:
            word_repr_sum = masked_states.sum(dim=1)
        repr_mean = word_repr_sum / (sep_pos - 1)[:, None]

        if layerwise_averaging:
            repr_mean = repr_mean.mean(dim=0)

    if input_type == "w1-s1|w2-s2":
        pooling_mask = batch["pooling_mask"]
        word_len = batch["pooling_mask"].sum(1)
        masked_states = hidden_states * pooling_mask[:, :, None]
        repr_mean = masked_states.sum(1) / word_len[:, None]

    if input_type == "w1-s1|w2-s2_mt":
        pooling_mask = batch["pooling_mask"]
        repr_mean = scatter_mean(hidden_states, pooling_mask, dim=1)

    if input_type == "w1-s1-w2-s2":
        pooling_mask = batch["pooling_mask"]
        mask = (pooling_mask != 0).long()
        offset = torch.arange(start=0, end=pooling_mask.shape[0] * 2, step=2)[
            :, None
        ].to(pooling_mask)
        offset_mask = mask * offset
        pool_mask = pooling_mask + offset_mask
        repr_mean = scatter_mean(hidden_states, pool_mask, dim=1).sum(0)[1:]

    return repr_mean
