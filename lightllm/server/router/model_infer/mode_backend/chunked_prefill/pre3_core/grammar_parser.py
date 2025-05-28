import argparse
import logging
import sys
from typing import List, Tuple
import copy

from lightllm.server.router.model_infer.mode_backend.chunked_prefill.pre3_core.core import (
    NT,
    T,
    compute_first,
    compute_graph,
)
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.pre3_core.dpda import Graph, LRGraph, DPDA

logger = logging.getLogger(__name__)

END_OF_ALTERNATE_MARKER = 0
END_OF_RULE_MARKER = 0
END_OF_GRAMMAR_MARKER = 0xFFFF
TO_BE_FILLED_MARKER = 0
REF_RULE_MARKER = 1
LITERAL_MARKER = 2


########################
# EBNF Grammar Parsing #
########################


class ParseState:
    def __init__(self):
        self.symbol_table = {}
        self.grammar_encoding = []  # old name: out_grammar

    def print(self, file=sys.stdout):
        print_grammar(file, self)

    def get_grammar(self, file=sys.stdout):
        return print_grammar(file, self)


def get_symbol_id(state: ParseState, symbol_name: str) -> int:
    if symbol_name not in state.symbol_table:
        state.symbol_table[symbol_name] = len(state.symbol_table)
    return state.symbol_table[symbol_name]


def generate_symbol_id(state: ParseState, base_name: str) -> int:
    next_id = len(state.symbol_table)
    state.symbol_table[base_name + "_" + str(next_id)] = next_id
    return next_id


def is_word_char(c: str) -> bool:
    """
    Check if a char is  a-z, A-Z, 0-9, -, _, i.e., chars allowed as rule names
    Returns:

    """
    return c.isalnum() or c == "-" or c == "_"


def hex_to_int(c: str) -> int:
    """
    Convert a hex char to int, c should be in the range of 0-9, a-f, A-F
    case insensitive
    Args:
        c:  a hex char
    Returns:
        int: the int value of the hex char
    """
    if c.isdigit():
        return int(c)
    elif "a" <= c.lower() <= "f":
        return ord(c.lower()) - ord("a") + 10
    return -1


def remove_leading_white_space(src, rm_leading_newline):
    """
    Skips over whitespace and comments in the input string.

    This function processes the input string, skipping over any spaces, tabs,
    and content following a '#' character, which denotes a comment. The parsing
    of a comment continues until the end of the line (denoted by newline characters
    '\r' or '\n'). If the 'rm_leading_newline' parameter is set to False, the function
    will stop processing and return the remaining string upon encountering a
    newline character, otherwise it will skip over newline characters as well.

    Parameters:
    src (str): The input string to be processed.
    rm_leading_newline (bool): A flag indicating whether encountering a newline character
                       should stop the parsing (False) or if it should be skipped (True).

    Returns:
    str: The remaining portion of the input string after skipping whitespace and comments.
    """
    pos = 0
    while pos < len(src) and (src[pos].isspace() or src[pos] == "#"):
        if src[pos] == "#":
            while pos < len(src) and src[pos] not in ("\r", "\n"):
                pos += 1
        else:
            if not rm_leading_newline and src[pos] in ("\r", "\n"):
                break
            pos += 1
    return src[pos:]


def parse_name(src) -> Tuple[str, str]:
    """
    parse the leading name from the input string
    Args:
        src:  the input grammar string

    Returns:
        name, remaining_src
    """
    pos = 0
    while pos < len(src) and is_word_char(src[pos]):
        pos += 1
    if pos == 0:
        raise RuntimeError("expecting name at " + src)
    return src[:pos], src[pos:]


def parse_char(src) -> Tuple[str, str]:
    """
    parse the leading char from the input string
    :param src:
    :return: char, remaining_src
    """

    # if we have a backslash, it's maybe an escape
    if src[0] == "\\":
        esc = src[1]
        if esc == "x":
            first = hex_to_int(src[2])
            if first > -1:
                second = hex_to_int(src[3])
                if second > -1:
                    return (first << 4) + second, src[4:]
            raise RuntimeError("expecting \\xNN at " + src)
        elif esc in ('"', "[", "]"):
            return esc, src[2:]
        elif esc == "r":
            return "\r", src[2:]
        elif esc == "n":
            return "\n", src[2:]
        elif esc == "t":
            return "\t", src[2:]
        elif esc == "\\":
            return "\\", src[2:]
        raise RuntimeError("unknown escape at " + src)
    elif src:
        return src[0], src[1:]
    raise RuntimeError("unexpected end of input")


def _parse_rhs_literal_string(src: str, outbuf: List[int]) -> str:
    assert src[0] == '"', f"rule should start with '\"', but got {src[0]}"
    remaining_src = src[1:]

    # advance until we get an end quote or run out of input
    while remaining_src and remaining_src[0] != '"':
        char, remaining_src = parse_char(remaining_src)
        outbuf.append(LITERAL_MARKER)
        outbuf.append(ord(char))
        outbuf.append(ord(char))

    # in case we ran out of input before finding the end quote
    if not remaining_src:
        raise RuntimeError(f"expecting an end quote at {src},but not found")

    # remove the end quote and return the remaining string
    return remaining_src[1:]


def _parse_rhs_char_ranges(src: str, outbuf: List[int]) -> str:
    assert src[0] == "[", f"rule should start with '[', but got {src[0]}"
    remaining_src = src[1:]
    start_idx = len(outbuf)
    # num chars in range - replaced at end of loop
    outbuf.append(TO_BE_FILLED_MARKER)
    while remaining_src and remaining_src[0] != "]":
        char, remaining_src = parse_char(remaining_src)

        outbuf.append(ord(char))
        if remaining_src[0] == "-" and remaining_src[1] != "]":
            endchar_pair, remaining_src = parse_char(remaining_src[1:])
            outbuf.append(ord(endchar_pair))
        else:
            # This is the case for enumerate, e.g., [0123456789], [abcdef]
            # Each char is considered as a range of itself, i.e., c-c
            outbuf.append(ord(char))
    if not remaining_src:
        raise RuntimeError(f"expecting an ] at {src},but not found, is the char range closed?")
    # replace num chars with actual
    outbuf[start_idx] = len(outbuf) - start_idx - 1
    return remaining_src[1:]


def _parse_rhs_symbol_reference(src: str, state: ParseState, outbuf: List[int]) -> str:
    assert is_word_char(src[0]), f"rule should start with a word char, but got {src[0]}"
    name, remaining_src = parse_name(src)
    ref_rule_id = get_symbol_id(state, name)
    outbuf.append(REF_RULE_MARKER)
    outbuf.append(ref_rule_id)
    return remaining_src


def _parse_rhs_grouping(remaining_src: str, state: ParseState, rule_name: str, outbuf: List[int]) -> str:
    assert remaining_src[0] == "(", f"rule should start with '(', but got {remaining_src[0]}"
    remaining_src = remove_leading_white_space(remaining_src[1:], True)
    # parse nested alternates into synthesized rule
    synthetic_rule_id = generate_symbol_id(state, rule_name)
    remaining_src = parse_rhs(state, remaining_src, rule_name, synthetic_rule_id, True)
    # output reference to synthesized rule
    outbuf.append(REF_RULE_MARKER)
    outbuf.append(synthetic_rule_id)

    if not remaining_src or remaining_src[0] != ")":
        raise RuntimeError("expecting ')' at " + remaining_src)
    return remaining_src[1:]


def _parse_rhs_repetition_operators(
    remaining_src: str,
    state: ParseState,
    rule_name: str,
    last_sym_start: int,
    outbuf: List[int],
) -> str:
    assert remaining_src[0] in (
        "*",
        "+",
        "?",
    ), f"rule should start with '*', '+', or '?', but got {remaining_src[0]}"
    out_grammar = state.grammar_encoding
    # last_sym_start = len(outbuf)

    # apply transformation to previous symbol (last_sym_start -
    # end) according to rewrite rules:
    # S* --> S' ::= S S' |
    # S+ --> S' ::= S S' | S
    # S? --> S' ::= S |
    sub_rule_id = generate_symbol_id(state, rule_name)
    out_grammar.append(sub_rule_id)
    sub_rule_offset = len(out_grammar)
    # placeholder for size of 1st alternate
    out_grammar.append(TO_BE_FILLED_MARKER)
    # add preceding symbol to generated rule
    out_grammar.extend(outbuf[last_sym_start:])
    if remaining_src[0] in ("*", "+"):
        # cause generated rule to recurse
        out_grammar.append(REF_RULE_MARKER)
        out_grammar.append(sub_rule_id)
    # apply actual size
    out_grammar[sub_rule_offset] = len(out_grammar) - sub_rule_offset
    # mark end of 1st alternate
    out_grammar.append(END_OF_ALTERNATE_MARKER)
    sub_rule_offset = len(out_grammar)
    # placeholder for size of 2nd alternate
    out_grammar.append(TO_BE_FILLED_MARKER)
    if remaining_src[0] == "+":
        # add preceding symbol as alternate only for '+'
        out_grammar.extend(outbuf[last_sym_start:])
    # apply actual size of 2nd alternate
    out_grammar[sub_rule_offset] = len(out_grammar) - sub_rule_offset
    # mark end of 2nd alternate, then end of rule
    out_grammar.append(END_OF_ALTERNATE_MARKER)
    out_grammar.append(END_OF_RULE_MARKER)

    # in original rule, replace previous symbol with reference to generated rule
    outbuf[last_sym_start:] = [REF_RULE_MARKER, sub_rule_id]
    return remaining_src[1:]


def parse_simple_rhs(state, rhs: str, rule_name: str, outbuf, is_nested):
    simple_rhs_offset = len(outbuf)

    # sequence size, will be replaced at end when known
    outbuf.append(TO_BE_FILLED_MARKER)

    last_sym_start = len(outbuf)
    remaining_rhs = rhs
    while remaining_rhs:
        if remaining_rhs[0] == '"':  # literal string
            # mark the start of the last symbol, for repetition operator
            last_sym_start = len(outbuf)
            remaining_rhs = _parse_rhs_literal_string(remaining_rhs, outbuf)
        elif remaining_rhs[0] == "[":  # char range(s)
            # mark the start of the last symbol, for repetition operator
            last_sym_start = len(outbuf)
            remaining_rhs = _parse_rhs_char_ranges(remaining_rhs, outbuf)
        elif is_word_char(remaining_rhs[0]):  # rule reference
            # mark the start of the last symbol, for repetition operator
            last_sym_start = len(outbuf)
            remaining_rhs = _parse_rhs_symbol_reference(remaining_rhs, state, outbuf)
        elif remaining_rhs[0] == "(":  # grouping
            # mark the start of the last symbol, for repetition operator
            last_sym_start = len(outbuf)
            remaining_rhs = _parse_rhs_grouping(remaining_rhs, state, rule_name, outbuf)
        elif remaining_rhs[0] in ("*", "+", "?"):  # repetition operator
            # No need to mark the start of the last symbol, because we already did it
            if len(outbuf) - simple_rhs_offset - 1 == 0:
                raise RuntimeError("expecting preceeding item to */+/? at " + remaining_rhs)
            remaining_rhs = _parse_rhs_repetition_operators(remaining_rhs, state, rule_name, last_sym_start, outbuf)
        else:
            # case for newline, i.e., end of rule
            assert remaining_rhs[0] in [
                "\n",
                "|",
                ")",
            ], f"rule should end with newline or '|', but got {remaining_rhs[0]}"
            # we break here so that we call parse_rule again to parse the next rule
            break
        # Here we do not rm newline deliberately so that we know the rhs is ended
        remaining_rhs = remove_leading_white_space(remaining_rhs, rm_leading_newline=is_nested)

    # apply actual size of this alternate sequence
    outbuf[simple_rhs_offset] = len(outbuf) - simple_rhs_offset
    # mark end of alternate
    outbuf.append(END_OF_ALTERNATE_MARKER)
    return remaining_rhs


def parse_rhs(state, rhs: str, rule_name, rule_id, is_nested):
    outbuf = []
    remaining_rhs = parse_simple_rhs(state, rhs, rule_name, outbuf, is_nested)
    while remaining_rhs and remaining_rhs[0] == "|":
        remaining_rhs = remove_leading_white_space(remaining_rhs[1:], True)
        remaining_rhs = parse_simple_rhs(state, remaining_rhs, rule_name, outbuf, is_nested)

    # Now we have finished parsing the rhs, we can add the rule to the grammar_encoding
    state.grammar_encoding.append(rule_id)
    state.grammar_encoding.extend(outbuf)
    state.grammar_encoding.append(END_OF_RULE_MARKER)
    return remaining_rhs


def parse_rule(state: ParseState, rule_text: str) -> str:
    name, remaining_rule_text = parse_name(rule_text)
    remaining_rule_text = remove_leading_white_space(remaining_rule_text, False)
    # check if the rule is already defined, TODO: what will happen if the rule is already defined?
    rule_id = get_symbol_id(state, name)

    if remaining_rule_text[:3] != "::=":
        raise RuntimeError("expecting ::= at " + remaining_rule_text)
    remaining_rule_text = remove_leading_white_space(remaining_rule_text[3:], True)

    remaining_rule_text = parse_rhs(state, remaining_rule_text, name, rule_id, False)

    if remaining_rule_text and remaining_rule_text[0] == "\r":
        remaining_rule_text = remaining_rule_text[2:] if remaining_rule_text[1] == "\n" else remaining_rule_text[1:]
    elif remaining_rule_text and remaining_rule_text[0] == "\n":
        remaining_rule_text = remaining_rule_text[1:]
    elif remaining_rule_text:
        raise RuntimeError("expecting newline or end at " + remaining_rule_text)
    return remove_leading_white_space(remaining_rule_text, True)


def parse_ebnf(grammar_text: str) -> ParseState:
    try:
        state = ParseState()
        remaining_grammar_text = remove_leading_white_space(grammar_text, True)
        last_grammar_repr = ""
        while remaining_grammar_text:
            if last_grammar_repr:
                last_parsed_rule_len = len(last_grammar_repr) - len(remaining_grammar_text)  # noqa
                # logger.debug(f"last_parsed_rule: {last_grammar_repr[:last_parsed_rule_len]}")
            last_grammar_repr = remaining_grammar_text
            remaining_grammar_text = parse_rule(state, remaining_grammar_text)
        state.grammar_encoding.append(END_OF_GRAMMAR_MARKER)
        return state
    except RuntimeError as err:
        logger.warning("error parsing grammar:", err)
        return ParseState()


###################################
# EBNF Grammar Parsing ends here  #
###################################


def break_grammar_into_rules(grammar_encoding: List[int]) -> List[List[int]]:
    offset = 0
    # we loop until we reach the end of the grammar_encoding
    rule_encodings = []
    i = 0
    while i < len(grammar_encoding) - 2:
        if grammar_encoding[i] == END_OF_ALTERNATE_MARKER and grammar_encoding[i + 1] == END_OF_RULE_MARKER:
            rule_encodings.append(grammar_encoding[offset : i + 2])
            offset = i + 2
            # skip the END_OF_RULE_MARKER
            # This is mandatory because if we do not skip the END_OF_RULE_MARKER
            # we fail in the case where the next rule has rule_id 0
            i += 1
        i += 1
    return rule_encodings


def break_rule_into_elements(rule_encoding: List[int]) -> List[List[int]]:
    # rule_id = rule_encoding.pop(0)
    end_of_rule_marker = rule_encoding.pop(-1)
    assert (
        end_of_rule_marker == END_OF_RULE_MARKER
    ), f"rule should end with {END_OF_RULE_MARKER}, but got {end_of_rule_marker}"

    offset = 0
    elements = []
    while offset < len(rule_encoding):
        element_size = rule_encoding[offset]
        assert (
            rule_encoding[offset + element_size] == END_OF_ALTERNATE_MARKER
        ), f"element should end with {END_OF_ALTERNATE_MARKER}, but got {rule_encoding[offset + element_size]}"
        elements.append(rule_encoding[offset : offset + element_size + 1])
        offset += element_size + 1
    return elements


def _print_annotated_grammar(file, grammar_encoding, symbol_id_names, index=0):
    rule = []
    rule_id = grammar_encoding[index]
    # print(f"<{index}>{symbol_id_names[rule_id]} ::=", end=" ", file=file)
    pos = index + 1
    while grammar_encoding[pos]:
        if pos - 1 > index:
            # print("|", end=" ", file=file)
            pass
        pos += 1  # sequence size, not needed here
        ref_list = []
        while grammar_encoding[pos]:
            if grammar_encoding[pos] == REF_RULE_MARKER:
                ref_rule_id = grammar_encoding[pos + 1]
                # print(
                #     f"<{pos}>{symbol_id_names[ref_rule_id]}",
                #     end=" ",
                #     file=file,
                # )
                ref_list.append(NT(f"{symbol_id_names[ref_rule_id]}"))
                pos += 2
            else:
                num_chars = grammar_encoding[pos]
                pos += 1
                all_string = ""
                for i in range(0, num_chars, 2):
                    for j in range(grammar_encoding[pos + i], grammar_encoding[pos + i + 1] + 1):
                        all_string += chr(j)
                    # print("{}-".format(chr(grammar_encoding[pos + i])), end="", file=file)
                    if i + 1 < num_chars:
                        # print(
                        #     "{}".format(chr(grammar_encoding[pos + i + 1])),
                        #     end="",
                        #     file=file,
                        # )
                        pass
                ref_list.append(T(all_string))
                # print("]", end=" ", file=file)
                pos += num_chars
        if len(ref_list) > 0:
            rule.append((NT(f"{symbol_id_names[rule_id]}"), ref_list))
        else:
            rule.append((NT(f"{symbol_id_names[rule_id]}"), [T("")]))  # T("") is a placeholder for \epsilon
        pos += 1
    # print(file=file)
    return pos + 1, rule


def print_grammar(file, state):
    grammar = []
    pos = 0
    symbol_id_names = {v: k for k, v in state.symbol_table.items()}
    # print("Grammar Rules:", file=file)
    while pos < len(state.grammar_encoding) and state.grammar_encoding[pos] != END_OF_GRAMMAR_MARKER:
        pos, rule = _print_annotated_grammar(file, state.grammar_encoding, symbol_id_names, pos)
        grammar.extend(rule)
    if pos > len(state.grammar_encoding):
        raise Warning(f"grammar_encoding is not ended with {END_OF_GRAMMAR_MARKER}")
    return grammar


def fix_grammar(grammar):
    # Split mutiple rules in terminals into single rules (e.g. A [ab] B -> A a B, A b B)
    # new_grammar = []
    # for i in range(len(grammar)):
    #     rule = grammar[i]
    #     lhs = rule[0]
    #     rhs = rule[1]
    #     new_rhs = []
    #     ok=True
    #     for j in range(len(rhs)):
    #         if isinstance(rhs[j], T) and len(rhs[j].value) > 1:
    #             ok=False
    #             for ch in rhs[j].value:
    #                 new_rhs = copy.deepcopy(rhs)
    #                 new_rhs[j] = T(ch)
    #                 new_grammar.append((lhs, new_rhs))
    #     if ok:
    #         new_grammar.append(rule)
    # grammar = new_grammar

    # Fix the epsilon rules
    while True:
        fix = True
        deleted_index = []
        for i in range(len(grammar)):
            rule = grammar[i]
            lhs = rule[0]
            rhs = rule[1]
            if len(rhs) == 1 and isinstance(rhs[0], T) and rhs[0].value == "":
                # print(f"rule: {rule}")
                fix = False
                deleted_index.append(i)
                # print("deleted:")
                for j in range(len(grammar)):
                    # if lhs exists in the other rules, we need to add the new rules
                    if i != j and lhs in grammar[j][1]:
                        new_rhs = copy.deepcopy(grammar[j][1])
                        new_rhs = [x for x in new_rhs if x != lhs]
                        if len(new_rhs) == 0:
                            new_rhs = [T("")]
                        grammar.append((grammar[j][0], new_rhs))
                        # print(grammar[j][0], new_rhs)
                break
        grammar = [grammar[i] for i in range(len(grammar)) if i not in deleted_index]
        if fix:
            break

    # Remove all rules if right hand side is empty
    deleted_index = []
    for i in range(len(grammar)):
        rule = grammar[i]
        lhs = rule[0]
        rhs = rule[1]
        if len(rhs) == 0:
            deleted_index.append(i)
    grammar = [grammar[i] for i in range(len(grammar)) if i not in deleted_index]

    return grammar


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Parse EBNF grammar files.")
    # parser.add_argument(
    #     "-g",
    #     "--grammar-file",
    #     nargs="?",
    #     default="examples/grammars/json.ebnf",
    #     help="Path to the grammar file (default: examples/grammars/json.ebnf)",
    # )

    # args = parser.parse_args()

    with open(
        "lightllm/server/router/model_infer/mode_backend/continues_batch/format_out/grammar/json_grammar.ebnf", "r"
    ) as file:
        input_text = file.read()

    parsed_grammar = parse_ebnf(input_text)
    grammar = parsed_grammar.get_grammar()
    grammar = fix_grammar(grammar)
    # print("Grammar in Our Format:")
    print(len(grammar))
    for i in grammar:
        print(i)
    # print(f"symbol_ids: \n{parsed_grammar.symbol_table}")

    # grammar = [
    #     (NT("S'"), [NT("S")]),
    #     (NT("S"), [NT("A")]),
    #     (NT("A"), [T("a"), T("a"), NT("A")]),
    #     (NT("A"), [T("a")]),
    #     (NT("A"), [T("c"), T("a"), NT("A")]),
    # ]

    graph = compute_graph(grammar=grammar, start_symbol="root")
    print("finish compute graph")
    graph.check_lr1()
    # graph.visit_print()
    lr_graph = LRGraph(graph)
    print("finish lr graph")
    dpda = DPDA(lr_graph=lr_graph)
    print("finish dpda")

    for in_str in ['{"value": true, "false": null, "null": [1,2,3]}']:
        try:
            dpda.accept(in_str)
            print(f"{in_str} accepted")
        except:
            print(f"{in_str} not accepted")
