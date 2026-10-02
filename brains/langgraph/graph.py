"""A five-stage LangGraph that yields after every batch of physical actions."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from brains.base import Decision
from .contracts import TaskState


class GraphState(TypedDict):
    task: TaskState
    actions: list[Decision]


def build_graph(stages):
    graph = StateGraph(GraphState)
    names = ("origin", "empty", "pending", "holding", "final")

    def node(name):
        def advance(value):
            task = value["task"]
            actions = stages.run(name, task)
            return {"task": task, "actions": actions}
        return advance

    def route(value):
        return END if value["actions"] else value["task"].phase

    for name in names:
        graph.add_node(name, node(name))
        graph.add_conditional_edges(name, route, {n: n for n in (*names, END)})
    graph.add_conditional_edges(START, lambda value: value["task"].phase, {n: n for n in names})
    return graph.compile()
