import streamlit as st
from langchain_core.messages import HumanMessage

from testing import build_graph

st.title("Indic - Indic RAG-based Translation")
st.write("""
A LangGraph application using fuzzy Indic-Indic RAG (Hindi, Odia, Tamil, Bengali, etc.) which contains Indic-Indic translation pairs instead of English as a pivot.
Enter any translation request in natural language with instructions on the language to translate to.
""")

# User input form
user_text = st.text_area("Enter translation request:", height=150)

if st.button("Translate"):
    if not user_text.strip():
        st.error("Please enter a valid translation request.")
    else:
        with st.spinner("Running graph:"):
            user_msg = HumanMessage(content=user_text)
            initial_state = {"messages": [user_msg]}

            graph = build_graph()
            result = graph.invoke(initial_state)

        st.subheader("Detected parameters")
        st.write(f"**Source language**: {result.get('source_lang')}")
        st.write(f"**Target language**: {result.get('target_lang')}")
        st.write(f"**Text to translate**: {result.get('query_for_retrieval')}")

        st.subheader("Retrieved Context (RAG)")
        ctx = result.get("retrieved_context") or ""
        if ctx:
            st.code(ctx)
        else:
            st.info("No RAG context retrieved, using direct translation path.")

        st.subheader("Final Translation")
        st.success(result.get("final_translation"))
