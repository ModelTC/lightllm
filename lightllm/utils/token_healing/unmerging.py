from typing import Optional, Sequence, Tuple


def seperate_last_unicode_char(token: bytes) -> Optional[Tuple[str, str]]:
    if not token:
        return None
    try:
        token_str = token.decode()
    except UnicodeDecodeError:
        return None
    if len(token_str) <= 1:
        return None
    return token_str[:-1], token_str[-1]


def build_unmerging_table(vocab: Sequence[bytes]):
    # NOTE: Here we assume only the last unicode character
    #       may be re-merged as the prefix of the newly healed token.

    # NOTE: It's fine to have multiple tokens with the same utf8
    #       representation. Token healing will choose the best token
    #       according to predicted probabilities.
    token_to_id_mapping = {token: i for i, token in enumerate(vocab)}

    def token_pair_to_id(pair: Optional[Tuple[str, str]]) -> Optional[Tuple[int, int]]:
        if not pair:
            return None
        u, v = map(str.encode, pair)
        if any(i not in token_to_id_mapping for i in (u, v)):
            return None
        u, v = map(token_to_id_mapping.__getitem__, (u, v))
        return u, v

    def unmerge(token_id) -> Optional[Tuple[int, int]]:
        return token_pair_to_id(seperate_last_unicode_char(vocab[token_id]))

    return list(map(unmerge, range(len(vocab))))
