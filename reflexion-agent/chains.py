from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import datetime
import os
from schema import AnswerQuestion, ReviseAnswer,parse_llm_response
from langchain_core.messages import HumanMessage


load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']

llm=ChatGroq(temperature=0, model_name="llama-3.1-8b-instant",groq_api_key=groq_api_key)


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda:datetime.datetime.now().isoformat(),
)

try:

    responder_prompt_template=actor_prompt_template.partial(
        first_instruction="Provide a detailed 30 word anwer"
    )
    responder_chain = responder_prompt_template | llm.bind_tools(
        tools=[AnswerQuestion],
        tool_choice="AnswerQuestion",
    )

    revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
    """

    revisor_prompt_template=actor_prompt_template.partial(
        first_instruction=revise_instructions
    )

    revisor_chain = revisor_prompt_template | llm.bind_tools(
        tools=[ReviseAnswer],
        tool_choice="ReviseAnswer",
    )

except Exception as e:
    print(f"Error: {e}")







