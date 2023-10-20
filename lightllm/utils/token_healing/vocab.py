import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

SP_TOKENIZER_BYTES_LITERAL = re.compile(r"(<0[xX]([0-9a-fA-F]{2})>){1,}")
SP_TOKENIZER_BYTES_REDUNDANT = re.compile(r"(<0[xX]|>){1,}")


def parse_sp_bytes_literal(token: str):
    return bytes(
        int(i, 16) for i in SP_TOKENIZER_BYTES_REDUNDANT.sub(" ", token).split()
    )


def hf_tokenizer_to_bytes_vocab(tokenizer: "PreTrainedTokenizerBase"):
    # ignore additional special tokens
    hf_vocab = tokenizer.get_vocab()
    vocab = [b""] * len(hf_vocab)

    for token, token_id in hf_vocab.items():
        if SP_TOKENIZER_BYTES_LITERAL.fullmatch(token):
            token_bytes = parse_sp_bytes_literal(token)
        else:
            token_bytes = tokenizer.decode(
                [tokenizer.bos_token_id, token_id, tokenizer.eos_token_id]
            )[len(tokenizer.bos_token) : -len(tokenizer.eos_token)].encode()

        vocab[token_id] = token_bytes

    # skip special tokens
    for special_id in tokenizer.all_special_ids:
        vocab[special_id] = b""

    return vocab
