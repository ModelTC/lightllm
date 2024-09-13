# 利用构建的 lr(1) 自动机，构建对应的dpda 自动机
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Any, Union, Dict, List, Tuple, Set, FrozenSet
from core import Item, T, NT, ItemLookAhead, ItemSet, Graph, Gen


@dataclass
class Edge:
    source_id: int
    transfer_input: Union[T, NT]
    dest_id: int
    is_loop_edge: bool = False


@dataclass
class LRGraph:
    origin_graph: Graph
    node_id_to_itemset: Dict[int, ItemSet] = None
    source_id_to_edge: Dict[int, Dict[Union[T, NT], Edge]] = None
    dest_id_to_edges: Dict[int, List[Edge]] = None
    # 保存能接收结束符的状态
    can_finished_node_id_set: Set[int] = None

    def __post_init__(self):
        self.node_id_to_itemset = {}
        self.source_id_to_edge = defaultdict(dict)
        self.dest_id_to_edges = defaultdict(list)
        self.can_finished_node_id_set = set()
        for node in self.origin_graph.graph_nodes:
            self.node_id_to_itemset[node.node_id] = node
            for input_t_or_nt, dest_node in node.edge_to_next.items():
                edge = Edge(node.node_id, input_t_or_nt, dest_node.node_id)
                self.source_id_to_edge[node.node_id][edge.transfer_input] = edge
                self.dest_id_to_edges[dest_node.node_id].append(edge)
            if node.can_finished():
                self.can_finished_node_id_set.add(node.node_id)


# DPDA 的转换边信息
@dataclass
class DPDAEdge:
    input_t: T
    pop: Tuple[int, ...]  # 与栈的方向相反
    push: Tuple[int, ...]  # 与栈的方向相同
    dest_node_id: int
    source_node_id: int

    def to_simple_str(self):
        ans = ""
        ans += f"t:{self.input_t.value}#pop:{self.pop}#push:{self.push}#s:{self.source_node_id}#e:{self.dest_node_id}"
        return ans


@dataclass
class DPDAEdgeMap:
    input_pop_to_edge: Dict[T, Dict[Tuple[int, ...], DPDAEdge]] = field(default_factory=lambda: defaultdict(dict))
    pop_input_to_edge: Dict[Tuple[int, ...], Dict[T, DPDAEdge]] = field(default_factory=lambda: defaultdict(dict))


@dataclass
class DPDA:
    lr_graph: LRGraph
    node_id_to_dpda_edges: Dict[int, DPDAEdgeMap] = None

    def add_dpadge(self, dpda_edge: DPDAEdge):
        self.node_id_to_dpda_edges[dpda_edge.source_node_id].input_pop_to_edge[dpda_edge.input_t][
            dpda_edge.pop
        ] = dpda_edge
        self.node_id_to_dpda_edges[dpda_edge.source_node_id].pop_input_to_edge[dpda_edge.pop][
            dpda_edge.input_t
        ] = dpda_edge

    def __repr__(self) -> str:
        ans = ""
        for source_id, edge_map in self.node_id_to_dpda_edges.items():
            ans += f"source id {source_id} ###################\n"
            for input_t, t_dict in edge_map.input_pop_to_edge.items():
                for pop, edge in t_dict.items():
                    ans += f"input {input_t} pop {pop} push {edge.push}, dest id {edge.dest_node_id} \n"
            ans += "##########################################\n"
        return ans

    def __post_init__(self):
        self.node_id_to_dpda_edges = defaultdict(DPDAEdgeMap)

        # 找到所有circles
        circles = []
        visit_stack = []
        visit_state = {}
        self.dfs_to_find_circle(0, visit_stack, visit_state, circles)

        # 处理找到的circle, 并做一些loop edge标记边, 和 添加一些dpda_edge
        self.handle_circles(circles)

        # 添加一些 不是 loop 边的 edge 到DPDA中
        for source_id, nt_or_t_to_edge in self.lr_graph.source_id_to_edge.items():
            for edge in nt_or_t_to_edge.values():
                if isinstance(edge.transfer_input, T) and not edge.is_loop_edge:
                    dpda_edge = DPDAEdge(
                        input_t=edge.transfer_input,
                        pop=(source_id,),
                        push=(source_id, edge.dest_id),
                        dest_node_id=edge.dest_id,
                        source_node_id=source_id,
                    )
                    self.add_dpadge(dpda_edge)

        # 递归处理可以处理的节点
        for graph_node in self.lr_graph.origin_graph.graph_nodes:
            graph_node.init_t_to_item_la()

            # 为每个 graph node 标记上，进入他的转移输入是什么
            tmp_edges = self.lr_graph.dest_id_to_edges[graph_node.node_id]
            if len(tmp_edges) == 0:
                assert graph_node.node_id == 0
            else:
                graph_node.into_t_or_nt = tmp_edges[0].transfer_input

        # 处理图中的包含可以规约节点的跳转
        self.recursion_update()
        return

    def recursion_update(self):
        for graph_node in self.lr_graph.origin_graph.graph_nodes:
            if isinstance(graph_node.into_t_or_nt, T):
                for t in graph_node.t_to_item_la.keys():
                    self._get_recurison(graph_node.node_id, t)
        return

    def _get_recurison(self, node_id: int, input_t: T) -> List[DPDAEdge]:
        tmp_ans_dict = self.node_id_to_dpda_edges[node_id].input_pop_to_edge[input_t]
        if tmp_ans_dict:
            return tmp_ans_dict.values()
        # 递归求解
        graph_node = self.lr_graph.node_id_to_itemset[node_id]
        pop_list = []
        self._dfs_recurison(node_id, node_id, graph_node.t_to_item_la[input_t], input_t, -1, pop_list)
        assert len(pop_list) == 0
        return self.node_id_to_dpda_edges[node_id].input_pop_to_edge[input_t].values()

    def _dfs_recurison(
        self,
        cur_node_id: int,
        iter_node_id: int,
        item_la: ItemLookAhead,
        input_t: T,
        back_index: int,
        pop_list: List[int],
    ):
        # back_index 从 -1 开始
        if -back_index > len(item_la.item.gen.gen_tuple):
            jump_to_node_id = self.lr_graph.source_id_to_edge[iter_node_id][item_la.item.gen.nt].dest_id
            # 判断是否有相关数据，没有就继续递归
            dpda_edges = self._get_recurison(jump_to_node_id, input_t)
            fake_dpdaedge = DPDAEdge(
                input_t=None,
                pop=(iter_node_id,),
                push=(
                    iter_node_id,
                    jump_to_node_id,
                ),
                dest_node_id=jump_to_node_id,
                source_node_id=iter_node_id,
            )  # 中间过渡边，没实际意义

            assert (
                len(dpda_edges) != 0
            ), f"cur_node_id {cur_node_id}, iter_node_id {iter_node_id}, \
            nt {item_la.item.gen.nt} jump_id {jump_to_node_id}, \
            input_t {input_t} back list {pop_list}"
            # print(f"iter start id {cur_node_id}")
            for t_jump_dpda_edge in dpda_edges:
                # print("id1", cur_node_id, "id2", t_jump_dpda_edge.dest_node_id, "jump_id", jump_to_node_id)
                new_dpda_edge = self.merge_dpda_edge(fake_dpdaedge, t_jump_dpda_edge)
                dest_dpda_edge = DPDAEdge(
                    input_t=input_t,
                    pop=tuple(pop_list) + new_dpda_edge.pop,
                    push=new_dpda_edge.push,
                    dest_node_id=t_jump_dpda_edge.dest_node_id,
                    source_node_id=cur_node_id,
                )

                # print("new edge", dest_dpda_edge)
                # print("old edge", t_jump_dpda_edge)
                self.add_dpadge(dest_dpda_edge)
            return

        pop_list.append(iter_node_id)
        for edge in lr_graph.dest_id_to_edges[iter_node_id]:
            assert edge.transfer_input == item_la.item.gen.gen_tuple[back_index]
            if not edge.is_loop_edge:
                self._dfs_recurison(cur_node_id, edge.source_id, item_la, input_t, back_index - 1, pop_list)

        pop_list.pop()
        return

    def merge_dpda_edge(self, first_edge: DPDAEdge, second_edge: DPDAEdge):
        assert first_edge.input_t is None
        if len(second_edge.pop) >= len(first_edge.push):
            push_size = len(first_edge.push)
            for i in range(push_size):
                assert first_edge.push[i] == second_edge.pop[push_size - i - 1]
            left_count = len(second_edge.pop) - len(first_edge.push)
            return DPDAEdge(
                input_t=second_edge.input_t,
                pop=(first_edge.pop + second_edge.pop[-left_count:]),
                push=second_edge.push,
                dest_node_id=second_edge.dest_node_id,
                source_node_id=first_edge.source_node_id,
            )
        else:
            print(first_edge.push, second_edge.pop)
            for i in range(len(second_edge.pop)):
                assert first_edge.push[-(i + 1)] == second_edge.pop[i]
            left_count = len(first_edge.push) - len(second_edge.pop)
            return DPDAEdge(
                input_t=second_edge.input_t,
                pop=first_edge.pop,
                push=first_edge.push[0:left_count] + second_edge.push,
                dest_node_id=second_edge.dest_node_id,
                source_node_id=first_edge.source_node_id,
            )

    def dfs_to_find_circle(
        self, cur_graph_id: int, visit_stack: List[int], visit_state_dict: Dict[int, int], ans_circles: List[List[int]]
    ):
        if cur_graph_id not in visit_state_dict:
            visit_stack.append(cur_graph_id)
            visit_state_dict[cur_graph_id] = len(visit_stack)  # 记录位置
            for edge in self.lr_graph.source_id_to_edge[cur_graph_id].values():
                self.dfs_to_find_circle(edge.dest_id, visit_stack, visit_state_dict, ans_circles)
            visit_stack.pop()
            del visit_state_dict[cur_graph_id]
            return
        else:
            origin_loc = visit_state_dict[cur_graph_id]
            tmp_ans = visit_stack[origin_loc - 1 :].copy()
            tmp_ans.append(cur_graph_id)
            ans_circles.append(tmp_ans)
            return

    def handle_circles(self, circles: List[List[int]]):
        for circle in circles:
            assert circle[0] == circle[-1]
            start_graph_node = self.lr_graph.node_id_to_itemset[circle[-2]]
            dest_graph_node = self.lr_graph.node_id_to_itemset[circle[-1]]
            input_t = self.lr_graph.dest_id_to_edges[dest_graph_node.node_id][0].transfer_input
            assert isinstance(input_t, T)
            self.lr_graph.source_id_to_edge[start_graph_node.node_id][input_t].is_loop_edge = True  # 标记这个边是loop 回旋边
            # 将已经知道的DPDAEdge 进行添加
            dpda_edge = DPDAEdge(
                input_t=input_t,
                pop=tuple(list(reversed(circle[0:-1]))),
                push=(circle[-1],),
                dest_node_id=circle[-1],
                source_node_id=start_graph_node.node_id,
            )

            self.add_dpadge(dpda_edge)
        return

    def to_mermaid(self):
        ans = "```mermaid\n"
        ans += "flowchart LR\n"
        for graph_node in self.lr_graph.origin_graph.graph_nodes:
            graph_info = graph_node.to_simple_str()
            ans += f'{graph_node.node_id}["' + graph_info + '"]\n'

        ans += "\n"
        for edge_maps in self.node_id_to_dpda_edges.values():
            for input_t_or_nt, pop_to_dpda_edge_dict in edge_maps.input_pop_to_edge.items():
                if isinstance(input_t_or_nt, T):
                    for pop, edge in pop_to_dpda_edge_dict.items():
                        edge_str = edge.to_simple_str().replace("(", "")
                        edge_str = edge_str.replace(")", "")
                        edge_str = edge_str.replace(" ", "")
                        ans += f"{edge.source_node_id} --> {edge_str} ---> {edge.dest_node_id}\n"

        ans += "```"
        return ans


if __name__ == "__main__":
    from core import compute_first, compute_graph

    grammar = [
        (NT("S'"), [NT("S")]),
        (NT("S"), [NT("A"), NT("B")]),
        (NT("A"), [T("a"), NT("A")]),
        (NT("A"), [T("a")]),
        (NT("B"), [T("b"), NT("B")]),
        (NT("B"), [T("b")]),
    ]
    # grammar = [
    #     (NT("S'"), [NT("S")]),
    #     (NT("S"), [NT("A"), NT("B")]),
    #     (NT("A"), [T("a"), T("a"), NT("A")]),
    #     (NT("A"), [T("a")]),
    #     (NT("B"), [T("b"), T("b"), NT("B")]),
    #     (NT("B"), [T("b")]),
    # ]
    ans = compute_first(grammar)
    print(ans)

    graph = compute_graph(grammar=grammar, start_symbol="S'")
    graph.visit_print()
    graph.check_lr1()
    graph_str = graph.to_mermaid()
    with open("mermaid.md", mode="+w") as file:
        file.write(graph_str)

    lr_graph = LRGraph(graph)
    dpda = DPDA(lr_graph=lr_graph)
    print(dpda)

    with open("mermaid1.md", mode="+w") as file:
        file.write(dpda.to_mermaid())
