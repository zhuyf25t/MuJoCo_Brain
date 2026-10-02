# 旧 LangGraph 归档

这些文件已过时，仅用于回溯；新流程见 [brains/langgraph.md](../../brains/langgraph.md)。当前没有调用旧实现的 `--brain` 参数。

- `brains/old_langgraph_brain.py`：归档前的旧 brain，包含当时的本地修改。
- `brains/langgraph_policy.py`、`brains/langgraph_memory.py`：旧提示词和 SQLite/checkpoint 存储代码。
- `brains/idea.md`、`brains/idea-design.md`、`brains/langgraph-design.md`：旧想法及设计。
- `docs/`：旧实现说明。
- `tests/langgraph_brain_tests.py`：旧测试原文，改为非自动发现的文件名，不再约束新的流程。

归档文件保留原文，包括当时的导入路径、相对链接和运行命令；它们不是当前可运行的接口。`brains` 下未发现实际的 `.db` / `.sqlite` 数据文件，已移除旧 LangGraph 的 Python 缓存。历史采集图片和轨迹仍保留在原位置。
