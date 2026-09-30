import os
from typing import Literal, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

from app.tools.rag_tool import tools


# 1. State Define karna
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    user_id: int

# Model initialize karna via OpenRouter
load_dotenv()

def get_model():
    api_key = os.getenv("OPENROUTER_API_KEY")
    return ChatOpenRouter(model="poolside/laguna-xs-2.1:free",
                api_key=api_key,
                max_tokens=1024)

# Agent Workflow
def create_agent_workflow(tools: list):
    model = get_model()
    model_with_tools = model.bind_tools(tools)

    # Agent Node
    async def agent_node(state: State):
        messages = state["messages"]
        thread_id = state["thread_id"]
        
        response = await model_with_tools.ainvoke(messages)
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_call["args"]["thread_id"] = thread_id
                
        return {"messages": [response]}

    # Tools Condition
    def tools_condition(state: State) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"

    # Graph construction
    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()

model = create_agent_workflow(tools)



# ======================================================================

async def get_thread_title(user_input: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = ChatOpenRouter(
        model="poolside/laguna-s-2.1:free",
        api_key=api_key,
        max_tokens=50
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Generate a short, catchy, and relevant thread title (maximum 4 to 5 words) based on the user's input. Do not include quotes or extra text."),
        ("human", "{input}")
    ])
    
    chain = prompt_template | model
    response = await chain.ainvoke({"input": user_input})
    
    return response.content.strip()