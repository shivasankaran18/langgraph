from langgraph.graph import MessageGraph,END
from chains import responder_chain, revisor_chain
from langchain_core.messages import BaseMessage, HumanMessage,ToolMessage
from tools import execute_tools
from typing import List


def response_node(state):
    return responder_chain.invoke({
        "messages": state
    })

def reflect_node(messages):
    return revisor_chain.invoke({
        "messages": messages
    })


def fn(state:List[BaseMessage])->str:
    cnt=0
    for item in state:
        if isinstance(item, ToolMessage):
            cnt+=1

    if cnt>2:
        return END
    return "reflect"

graph=MessageGraph()
graph.add_node("generate", response_node)
graph.add_node("reflect", reflect_node)
graph.add_node("execute_tools", execute_tools)

graph.add_edge("generate","execute_tools")
graph.add_edge("execute_tools","reflect")
graph.add_conditional_edges("reflect",fn)

graph.set_entry_point("generate")


app=graph.compile()
print(app.get_graph().draw_mermaid())

response=app.invoke(HumanMessage(content="AI Agents taking over content creation"))


print(response[-1].tool_calls[0]["args"]["answer"])


