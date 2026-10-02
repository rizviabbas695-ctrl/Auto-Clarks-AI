import os
import streamlit as st
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

BOT_NAME = "Car Agency Assistant"

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title=BOT_NAME, page_icon="🚗")
st.title("🚗 " + BOT_NAME)
st.caption("Content, Sales, Task planning, ya Accounts — kuch bhi poochho")


class FlowState(TypedDict):
    question: str
    category: str
    answer: str


class QuestionCategory(BaseModel):
    category: Literal['content', 'sales', 'task', 'accounts'] = Field(
        default="content",
        description="User ke sawaal ki category"
    )


@st.cache_resource
def build_graph():
    llm = ChatGroq(model="openai/gpt-oss-20b")

    def check_question_category(state: FlowState) -> FlowState:
        st_llm = llm.with_structured_output(QuestionCategory)
        res = st_llm.invoke(
            f"""Categorize this question into one of: content, sales, task, accounts.

- content: social media post ideas, captions, prompts, posting timing
- sales: how to talk to dealers or customers, negotiation scripts
- task: how many calls or meetings to do, daily planning
- accounts: commission earned, expenses, sales percentage, money tracking

Question: {state['question']}"""
        )
        state["category"] = res.category
        return state

    def route(state: FlowState) -> Literal["content", "sales", "task", "accounts"]:
        return state["category"]

    def content_node(state: FlowState) -> FlowState:
        prompt = f"""You are a social media content manager for a used-car sales agency
that markets 2nd-hand cars for dealers in Karol Bagh via Instagram/Facebook.
Help with captions, post ideas, content prompts, or best posting times.
Keep it practical and ready to use.

Request: {state['question']}"""
        state["answer"] = llm.invoke(prompt).content
        return state

    def sales_node(state: FlowState) -> FlowState:
        prompt = f"""You are a sales manager for a used-car agency. The agency earns
commission (around 5-7%) by connecting Karol Bagh car dealers with buyers via
social media leads. Give practical scripts or advice for talking to dealers
(negotiating commission) or customers (building trust, closing the sale).

Request: {state['question']}"""
        state["answer"] = llm.invoke(prompt).content
        return state

    def task_node(state: FlowState) -> FlowState:
        prompt = f"""You are a task manager for a 2-person used-car agency working
in Karol Bagh. Help plan daily/weekly targets — how many dealers to call,
how many in-person meetings to do, and how to prioritize follow-ups.
Be specific and realistic for a small team.

Request: {state['question']}"""
        state["answer"] = llm.invoke(prompt).content
        return state

    def accounts_node(state: FlowState) -> FlowState:
        prompt = f"""You are an accountant for a used-car agency that earns commission
on car sales (e.g. 7% asked, often settles at 5-6%). Help track commission
earned, expenses, and monthly sale percentages. If the user gives numbers,
calculate clearly. If not, explain what to track and how.

Request: {state['question']}"""
        state["answer"] = llm.invoke(prompt).content
        return state

    graph = StateGraph(FlowState)
    graph.add_node("check_question_category", check_question_category)
    graph.add_node("content", content_node)
    graph.add_node("sales", sales_node)
    graph.add_node("task", task_node)
    graph.add_node("accounts", accounts_node)

    graph.add_edge(START, "check_question_category")
    graph.add_conditional_edges("check_question_category", route)
    graph.add_edge("content", END)
    graph.add_edge("sales", END)
    graph.add_edge("task", END)
    graph.add_edge("accounts", END)

    return graph.compile()


finalGraph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

q = st.chat_input("Type your question...")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = finalGraph.invoke({"question": q, "category": "", "answer": ""})
            a = result["answer"]
            category = result["category"]
        st.caption(f"Routed to: {category}")
        st.write(a)
    st.session_state.messages.append({"role": "assistant", "content": a})
