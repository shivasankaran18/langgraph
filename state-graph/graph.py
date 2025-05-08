from typing import TypedDict,Annotated,List
from langgraph.graph import StateGraph,END,add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_community.tools import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os
load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']

conn=sqlite3.connect("db",check_same_thread=False)

memory=SqliteSaver(conn)

class State(TypedDict):
    messages:Annotated[List,add_messages]

search=TavilySearchResults(api_key=os.environ['TAVILY_API_KEY'])



tools=[search]
toolnode=ToolNode(tools)

llm=ChatGroq(temperature=0, model_name="Gemma2-9b-It",groq_api_key=groq_api_key)
llm_with_tools = llm.bind_tools(tools=tools)

def node(state: State):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])], 
    }

def tools_router(state: State):
    last_message = state["messages"][-1]


    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END
    

tool_node = ToolNode(tools=tools)

graph = StateGraph(State)

graph.add_node("chatbot", node)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "chatbot")

app = graph.compile(checkpointer=memory)

config={
    "configurable":{
        "thread_id":1
    }
}

while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        },config=config)

        # for item in result['messages']:
         
        #     if isinstance(item, ToolMessage):
        #         print(f"Tool: {item.content}")
        print(result)




    