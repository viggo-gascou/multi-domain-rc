import os
import torch
import torch.nn as nn
import transformers
from ..rc_types import Experiment
from dotenv import load_dotenv

load_dotenv()

#
# Embeddings Base Class
#


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = None

    def __repr__(self):
        return f"<{self.__class__.__name__}: dim={self.emb_dim}>"


class TransformerEmbeddings(Embeddings):
    def __init__(self, lm_name, experiment_type):
        super().__init__()
        self.experiment_type = experiment_type
        # load transformer
        self._tok = transformers.AutoTokenizer.from_pretrained(
            lm_name, use_fast=True, add_prefix_space=True
        )
        self._lm = transformers.AutoModel.from_pretrained(lm_name, return_dict=True)
        config = self._lm.config
        if self.experiment_type == Experiment.dataset_embeddings:
            self.dataset_embeddings = nn.Embedding(
                config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
            )

        # move model to GPU if available
        if torch.cuda.is_available():
            self._lm.to(torch.device("cuda"))

        # add special tokens
        ner_labels = os.getenv("ENTITY_LABELS").split()
        domain_labels = os.getenv("DOMAINS").split()
        self._tok.add_special_tokens(
            {
                "additional_special_tokens": self.get_special_tokens(
                    ner_labels, domain_labels
                )
            }
        )
        self._lm.resize_token_embeddings(len(self._tok))

        # public variables
        self.emb_dim = self._lm.config.hidden_size

    def get_special_tokens(self, ner_labels, domain_labels):
        special_tokens = []

        for label in ner_labels:
            special_tokens.append(f"<E1:{label}>")
            special_tokens.append(f"</E1:{label}>")
            special_tokens.append(f"<E2:{label}>")
            special_tokens.append(f"</E2:{label}>")
            special_tokens.extend(["<E1>", "</E1>", "<E2>", "</E2>"])

        if self.experiment_type == Experiment.special_token:
            for domain in domain_labels:
                special_tokens.append(f"[{domain.upper()}]")

        return special_tokens

    def embed(self, sentences):
        embeddings = []
        emb_words, _ = self.forward(sentences)
        # gather non-padding embeddings per sentence into list
        for sidx in range(len(sentences)):
            embeddings.append(emb_words[sidx, : len(sentences[sidx]), :].cpu().numpy())
        return embeddings

    def forward(self, sentences, domains):
        tok_sentences = self.tokenize(sentences)
        model_inputs = {
            k: tok_sentences[k]
            for k in ["input_ids", "token_type_ids", "attention_mask"]
            if k in tok_sentences
        }

        if self.experiment_type == Experiment.dataset_embeddings:
            dataset_to_id = {k: i for i, k in enumerate(os.getenv("DOMAINS").split())}
            data_ids = torch.zeros_like(model_inputs["input_ids"])
            data_ids[:] = torch.tensor(
                [dataset_to_id[domain] for domain in domains]
            ).reshape(-1, 1)
            word_embeds = self._lm.embeddings.word_embeddings(model_inputs["input_ids"])
            dataset_embeds = self.dataset_embeddings(data_ids).to(self._lm.device)
            model_inputs["inputs_embeds"] = word_embeds + dataset_embeds
            model_inputs.pop("input_ids")

        # perform embedding forward pass
        model_outputs = self._lm(**model_inputs, output_hidden_states=True)

        # extract embeddings from relevant layer
        hidden_states = (
            model_outputs.hidden_states
        )  # tuple(num_layers * (batch_size, max_len, hidden_dim))
        emb_pieces = hidden_states[-1]  # batch_size, max_len, hidden_dim

        # return model-specific tokenization
        return emb_pieces, tok_sentences["attention_mask"], tok_sentences.encodings

    def tokenize(self, sentences):
        # tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]], special_tokens_mask: [[]]}
        tok_sentences = self._tok(
            [sentence.split(" ") for sentence in sentences],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # move input to GPU (if available)
        if torch.cuda.is_available():
            tok_sentences = tok_sentences.to(torch.device("cuda"))

        return tok_sentences


#
# Pooling Function
#


def get_marker_embeddings(token_embeddings, encodings, ent1, ent2):
    if torch.cuda.is_available():
        start_markers = torch.Tensor().to(torch.device("cuda"))
    else:
        start_markers = torch.Tensor()

    for embedding, word_id in zip(token_embeddings, encodings.word_ids):
        if (word_id == ent1) or (word_id == ent2):
            start_markers = torch.cat([start_markers, embedding])
    return start_markers
