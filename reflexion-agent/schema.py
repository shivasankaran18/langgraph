from pydantic import BaseModel, Field
from typing import List


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous") 

class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(
        description="~250 word detailed answer to the question.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
    reflection: Reflection = Field(
        description="Your reflection on the initial answer.")
    
class ReviseAnswer(BaseModel):
    """Revise the answer based on the critique."""
    
    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )



def parse_llm_response(response):
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_call = response.tool_calls[0]

        print(tool_call)
        parsed_response = AnswerQuestion(
            answer=tool_call.get('args').get('answer'),
            reflection=Reflection(
                missing=tool_call.get('args').get('reflection', {}).get('missing', ''),
                superfluous=tool_call.get('args').get('reflection', {}).get('superfluous', '')
            ),
            search_queries=tool_call.get('args').get('search_queries', [])
        )
        
        return parsed_response
    else:
        raise ValueError("No tool calls found in the response")
