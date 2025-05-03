from langchain_community.tools import TavilySearchResults
import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from typing import List
from langchain_community.tools import TavilySearchResults
import json
load_dotenv()



search = TavilySearchResults(api_key=os.environ['TAVILY_API_KEY'],max_results=5)


def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message: AIMessage = state[-1]
    
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []

    tool_messages = []
    
    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])
            
            query_results = {}
            for query in search_queries:
                result = search.invoke(query)
                query_results[query] = result
          
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=call_id
                )
            )
    
    return tool_messages
