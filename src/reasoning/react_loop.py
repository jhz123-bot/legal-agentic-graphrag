from typing import Any, Dict, List


def plan_react_actions(query: str, route_decision: Dict[str, Any], max_steps: int = 3) -> List[Dict[str, str]]:
    preferred_tools = route_decision.get("preferred_tools", ["graph_search"])
    actions: List[Dict[str, str]] = []
    for tool_name in preferred_tools[:max_steps]:
        thought = f"需要使用工具 {tool_name} 获取与问题相关证据。"
        actions.append(
            {
                "thought": thought,
                "action": tool_name,
                "action_input": query,
            }
        )
    return actions


def react_loop_node(state: Dict[str, Any]) -> Dict[str, Any]:
    decision = state.get("router_decision", {})
    actions = plan_react_actions(state["user_query"], decision, max_steps=3)
    logs = list(state.get("logs", []))
    logs.append(f"react_loop: planned_actions={len(actions)}")
    return {"react_plan": actions, "logs": logs}


def make_tool_calling_node(tools: Dict[str, Any]):
    def tool_calling_node(state: Dict[str, Any]) -> Dict[str, Any]:
        trace = []
        tool_results = []
        for item in state.get("react_plan", []):
            action = item["action"]
            thought = item["thought"]
            query = item["action_input"]
            tool = tools.get(action)
            if tool is None:
                observation = f"工具 {action} 不可用"
                result = {"tool": action, "summary": observation}
            else:
                result = tool.run(query)
                observation = result.get("summary", "ok")
            trace.append(
                {
                    "thought": thought,
                    "action": action,
                    "observation": observation,
                }
            )
            tool_results.append(result)

        logs = list(state.get("logs", []))
        logs.append(f"tool_calling: executed={len(trace)}")
        return {"react_trace": trace, "tool_results": tool_results, "logs": logs}

    return tool_calling_node
