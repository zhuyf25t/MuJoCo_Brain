"""START -> model -> tools -> model or END; fatal exceptions propagate."""

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from .state import VistaState, validate_reply


def after_tools(state: VistaState):
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        calls = validate_reply(last)
        if len(calls) != 1 or calls[0]["name"] not in ("play", "finish"):
            raise RuntimeError("VISTA 工具节点未给本地调用返回结果，程序终止。")
        return END
    return "model"


def build_graph(model_node, tools_node):
    graph = StateGraph(VistaState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "model")
    graph.add_edge("model", "tools")
    graph.add_conditional_edges("tools", after_tools, {"model": "model", END: END})
    return graph.compile()
