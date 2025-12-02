import streamlit as st
from langchain_core.messages import HumanMessage

from main import build_graph   

st.set_page_config(
    page_title="Indic-Indic NLP-Powered RAG Translation Agent",
    layout="wide",
)

st.sidebar.title("Indic-Indic RAG Translation")
st.sidebar.markdown("""
This application performs direct translation between Indic languages using LangGraph and fuzzy-aligned Indic-Indic sentence pairs that were created without the usage of English as a pivot, using the Samanantar dataset.
""")

st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
Enter a translation command in natural language.  
Examples:

- मैं इसका उड़िया में अनुवाद करना चाहता हूँ: मैं आकाश हूँ और मुझे वनीला आइसक्रीम बहुत पसंद है 
- Translate this to Tamil: ਮੇਰਾ ਨਾਮ ਆਕਾਸ਼ ਹੈ ਅਤੇ ਮੈਨੂੰ ਵਨੀਲਾ ਆਈਸ ਕਰੀਮ ਬਹੁਤ ਪਸੰਦ ਹੈ।         
""")

st.sidebar.markdown("---")
st.sidebar.caption("Built for the MAT496 Capstone Project.")

st.markdown("""
# Direct Indic - Indic Translation using RAG & LangGraph
A retrieval-augmented translation system that avoids English as a pivot language by using fuzzy-aligned Indic sentence pairs. Supported languages: English, Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu

""")
st.markdown("""Translation datasets already generated: Bengali->Oriya, Hindi->Assamese, Hindi->Bengali, Hindi->Gujarati, Hindi->Kannada, Hindi->Malayalam, Hindi->Marathi, Hindi->Oriya, Hindi->Punjabi, Hindi->Tamil, Hindi->Telugu, Oriya->Hindi, Tamil->Hindi, Tamil->Malayalam, Tamil->Punjabi
""")
st.markdown("""*For other use cases, for eg. Punjabi->Malayalam, a dataset will be created while the graph is running and then proceed to translation. This may take time.
For languages not listed above, direct translation will be utilized.*""")
st.markdown("### Enter Translation Request")

user_text = st.text_area(
    "Type your translation request here",
    height=160,
    placeholder="Example: मैं इसका उड़िया में अनुवाद करना चाहता हूँ: मैं आकाश हूँ और मुझे वनीला आइसक्रीम बहुत पसंद है"
)

submit = st.button("Translate")

if submit:
    if not user_text.strip():
        st.warning("Please enter some text.")
        st.stop()

    st.markdown("### Processing:")
    with st.spinner("Running graph:"):
        graph = build_graph()  
        user_msg = HumanMessage(content=user_text.strip())
        initial_state = {"messages": [user_msg]}
        result = graph.invoke(initial_state)

    st.markdown("## Translation Results")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Detected Source Language:**", result.get("source_lang"))
        st.write("**Detected Target Language:**", result.get("target_lang"))

    with col2:
        st.write("**Text to Translate:**")
        st.info(result.get("query_for_retrieval"))

    st.markdown("---")

    retrieved = result.get("retrieved_context")

    if retrieved:
        st.markdown("### Retrieved Context (RAG)")
        st.caption("These aligned Indic-Indic sentence pairs guided the translation.")

        with st.expander("View All Retrieved Examples"):
            st.text(retrieved)
    else:
        st.markdown("### Retrieved Context (RAG)")
        st.info("No context retrieved. Direct translation used.")

    st.markdown("---")

    st.markdown("### Final Translation")
    st.success(result.get("final_translation"))

    st.markdown("---")
    st.caption("End of translation.")
