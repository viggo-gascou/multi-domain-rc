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
        self.dataset_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # move model to GPU if available
        if torch.cuda.is_available():
            self._lm.to(torch.device("cuda"))

        # add special tokens
        ner_labels = os.getenv(f"ENTITY_LABELS").split()
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
        emb_words, att_words = self.forward(sentences)
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
            data_ids = self.compute_data_ids(
                sentences, model_inputs["input_ids"], domains
            )
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

    def compute_offsets(self, sentences, max_len):
        offsets_list = []
        for sentence in sentences:
            token_count = 0
            offsets = []
            string_tokens = sentence.split(" ")
            for token_string in string_tokens:
                wordpieces = self._tok.encode_plus(
                    token_string,
                    add_special_tokens=False,
                    return_tensors=None,
                    return_offsets_mapping=False,
                    return_attention_mask=False,
                )
                wp_ids = wordpieces["input_ids"]
                if len(wp_ids) > 0:
                    offsets.append((token_count, token_count + len(wp_ids) - 1))
                    token_count += len(wp_ids)
                else:
                    offsets.append(None)
            offsets = [x if x is not None else (-1, -1) for x in offsets]

            # Truncate/pad offsets
            offsets = offsets[:max_len]
            pad_length = max_len - len(offsets)
            values_to_pad = [(0, 0)] * pad_length
            offsets = offsets + values_to_pad

            offsets_list.append(offsets)
        return torch.Tensor(offsets_list)

    def compute_dataset_ids(self, sentences, domains):
        dataset_to_id = {k: i for i, k in enumerate(os.getenv("DOMAINS").split())}
        dataset_ids_list = [
            [dataset_to_id[domain] for _ in sentence.split(" ")]
            for sentence, domain in zip(sentences, domains)
        ]
        max_len = len(max(dataset_ids_list, key=len))
        for i, dataset_ids in enumerate(dataset_ids_list):
            dataset_ids_list[i] = dataset_ids[:max_len]
            pad_length = max_len - len(dataset_ids_list[i])
            values_to_pad = [0] * pad_length
            dataset_ids_list[i] = dataset_ids_list[i] + values_to_pad
        return torch.Tensor(dataset_ids_list)

    def compute_data_ids(self, sentences, input_ids, domains):
        offsets = self.compute_offsets(sentences, input_ids.shape[1])
        wordpiece_sizes = []
        for sent_idx in range(len(offsets)):
            wordpiece_sizes.append([])
            for word_idx in range(len(offsets[sent_idx])):
                if offsets[sent_idx][word_idx][0] == 0:
                    continue
                wordpiece_sizes[-1].append(
                    int(
                        offsets[sent_idx][word_idx][1]
                        - offsets[sent_idx][word_idx][0]
                        + 1
                    )
                )
        dataset_ids = self.compute_dataset_ids(sentences, domains)
        data_ids = torch.zeros_like(input_ids)
        for sent_idx in range(len(wordpiece_sizes)):
            piece_idx = 0
            for word_idx in range(
                min(len(wordpiece_sizes[sent_idx]), len(dataset_ids[sent_idx]))
            ):
                for _ in range(wordpiece_sizes[sent_idx][word_idx]):
                    if piece_idx >= len(data_ids[sent_idx]):
                        continue
                    data_ids[sent_idx][piece_idx] = dataset_ids[sent_idx][word_idx]
                    piece_idx += 1
        return data_ids


#
# Pooling Function
#


def get_marker_embeddings(token_embeddings, encodings, ent1, ent2):
    if torch.cuda.is_available():
        start_markers = torch.Tensor().to(torch.device("cuda"))
    else:
        start_markers = torch.Tensor()

    for embedding, word_id in zip(token_embeddings, encodings.word_ids):
        if word_id == ent1:
            start_markers = torch.cat([start_markers, embedding])
        elif word_id == ent2:
            start_markers = torch.cat([start_markers, embedding])
    return start_markers
