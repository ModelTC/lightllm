from core import compute_first, compute_graph
from dpda import NT, LRGraph, Graph, DPDA
from dpda import T as TT


def create_grammer():
    # add = TT("a")
    # sub = TT("-")
    mul = TT("c")
    # div = TT("/")
    lparen = TT("l")
    rparen = TT("r")
    num = [TT("1"), TT("2"), TT("3")]

    E = NT("E")
    T = NT("T")
    F = NT("F")
    # NUM = NT("NUM")

    grammar = [
        (NT("S'"), [E]),
        # (E, [E, add, T]),
        # (E, [E, sub, T]),
        (E, [T]),
        (T, [T, mul, F]),
        # (T, [T, div, F]),
        (T, [F]),
        (F, [lparen, E, rparen]),
        (F, [num[0]]),
        # (F, [NUM]),
        # (NUM, [num[0]]),
        # (NUM, [num[1]]),
        # (NUM, [num[2]]),
    ]
    return grammar


grammar = create_grammer()

ans = compute_first(grammar)
print(ans)

graph = compute_graph(grammar=grammar, start_symbol="S'")

graph_str = graph.to_mermaid()
with open("mermaid.md", mode="+w") as file:
    file.write(graph_str)

graph.visit_print()
graph.check_lr1()
graph_str = graph.to_mermaid()
with open("mermaid.md", mode="+w") as file:
    file.write(graph_str)

lr_graph = LRGraph(graph)
dpda = DPDA(lr_graph=lr_graph)
print(dpda)

dpda.remove_no_input_node_to_edges()

with open("mermaid1.md", mode="+w") as file:
    file.write(dpda.to_mermaid())

for in_str in ["1a1", "1a1c1", "1a1a1a1c1c1", "1cl1c1r", "aaaaabbbb"]:
    try:
        dpda.accept(in_str)
        print(f"{in_str} accepted")
    except:
        print(f"{in_str} not accepted")
