import operator
from typing import TypedDict, Annotated, Sequence

from pydantic import BaseModel, Field
from IPython.display import Image, display
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from datasets import load_dataset
import pandas as pd
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import itertools

_INDIC_VECTOR_STORES = {}

class LanguageIntent(BaseModel):
    """data extracted from the user's input for translation."""
    source_lang_code: str = Field(description="The language code (e.g., 'hindi') of the original user query")
    target_lang_code: str = Field(description="The language code (e.g., 'marathi') of the desired translation output")
    query_for_retrieval: str = Field(description="The phrase from the user's input that needs translation")

class TranslationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    source_lang: str           
    target_lang: str           
    query_for_retrieval: str   

    retrieved_context: str     # top search results from ChromaDB

    final_translation: str     

    tool_call_result: str      

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embedding_model = None
vector_store = None

def build_indic_indic_rag_fuzzy(
    src_indic: str,
    tgt_indic: str,
    sample_per_lang: int = 30000,
    embed_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    persist_root: str = "./chroma_indic_pairs_fuzzy",
    sim_threshold: float = 0.80
):
    """
    Since Samanantar only contains English - Indic language pairs, we're finding the nearest English sentence in two datasets and using semantic similarities (say 80% similar) we're creating aligned pairs of Indic-Indic languages using fuzzy semantic alignment
    Creates a vector store embedding ONLY the source Indic sentences.
    """

    print(f"\nloading samanantar splits: {src_indic}, {tgt_indic}")

    # loading Samanantar splits
    ds_src = load_dataset("ai4bharat/samanantar", src_indic, split="train")
    ds_tgt = load_dataset("ai4bharat/samanantar", tgt_indic, split="train")

    # Subsample for efficiency
    if sample_per_lang < len(ds_src):
        idx_src = np.random.choice(len(ds_src), size=sample_per_lang, replace=False)
        ds_src = ds_src.select(idx_src)

    if sample_per_lang < len(ds_tgt):
        idx_tgt = np.random.choice(len(ds_tgt), size=sample_per_lang, replace=False)
        ds_tgt = ds_tgt.select(idx_tgt)

    print("Sample sizes:", len(ds_src), len(ds_tgt))

    src_df = pd.DataFrame(ds_src, columns=["src", "tgt"])
    tgt_df = pd.DataFrame(ds_tgt, columns=["src", "tgt"])

    src_df.rename(columns={"src": "en_src", "tgt": "src_indic"}, inplace=True)
    tgt_df.rename(columns={"src": "en_tgt", "tgt": "tgt_indic"}, inplace=True)

    embedder = SentenceTransformerEmbeddings(model_name=embed_model_name)

    en_src_list = src_df["en_src"].tolist()
    en_tgt_list = tgt_df["en_tgt"].tolist()

    en_src_emb = np.array(embedder.embed_documents(en_src_list))
    en_tgt_emb = np.array(embedder.embed_documents(en_tgt_list))

    en_src_emb /= np.linalg.norm(en_src_emb, axis=1, keepdims=True) #normalization
    en_tgt_emb /= np.linalg.norm(en_tgt_emb, axis=1, keepdims=True)

    # nearest neighbor search for english alignment
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="auto")
    nn.fit(en_tgt_emb)

    distances, indices = nn.kneighbors(en_src_emb)
    sims = 1 - distances[:, 0]

    print(f"Average similarity: {sims.mean():.4f}")
    print(f"Pairs above threshold {sim_threshold}: {np.sum(sims >= sim_threshold)}")

    # build aligned Indic-Indic rows
    valid_mask = sims >= sim_threshold
    valid_src_idx = np.where(valid_mask)[0]
    valid_tgt_idx = indices[valid_mask].flatten()

    aligned_df = pd.DataFrame({
        "src_indic": src_df.iloc[valid_src_idx]["src_indic"].values,
        "tgt_indic": tgt_df.iloc[valid_tgt_idx]["tgt_indic"].values,
        "sim": sims[valid_mask],
        "en_src": src_df.iloc[valid_src_idx]["en_src"].values,
        "en_tgt": tgt_df.iloc[valid_tgt_idx]["en_tgt"].values
    })
    print("Aligned Indic–Indic pairs:", aligned_df.shape)

    if aligned_df.empty:
        raise ValueError("No fuzzy matches found")

    metadatas = [
        {
            "src_indic": src,
            "tgt_indic": tgt,
            "sim": float(sim),
            "en_src": en_s,
            "en_tgt": en_t
        }
        for src, tgt, sim, en_s, en_t in zip(
            aligned_df["src_indic"],
            aligned_df["tgt_indic"],
            aligned_df["sim"],
            aligned_df["en_src"],
            aligned_df["en_tgt"],
        )
    ]

    texts = aligned_df["src_indic"].astype(str).tolist()
    ids = [str(i) for i in range(len(texts))]

    persist_dir = f"{persist_root}/{src_indic}_{tgt_indic}"
    collection_name = f"indic_fuzzy_{src_indic}_{tgt_indic}"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_dir
    )
    # add to chroma
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Stored {len(texts)} Indic-Indic aligned pairs at {persist_dir}")

    return vector_store, aligned_df

def load_or_create_vector_store(src: str, tgt: str, embed_model_name: str):
    """
    Loads the persisted chroma db for the given language pair based on whether it's already been created.
    Otherwise builds a new RAG db (fuzzy)
    Returns created chroma object.
    """
    persist_dir = f"./chroma_indic_pairs_fuzzy/{src}_{tgt}"
    collection_name = f"indic_fuzzy_{src}_{tgt}"

    try:
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=SentenceTransformerEmbeddings(model_name=embed_model_name),
            persist_directory=persist_dir
        )

        if vs._collection.count() > 0:
            print(f"Loaded existing vector store from disk for {src}->{tgt}")
            return vs
        else:
            print(f"Found a directory but it is empty. rebuilding DB: {persist_dir}")

    except Exception as e:
        print(f"No existing store {e}. Building new DB.")

    print(f"Building new DB for {src}->{tgt}")
    vs, _ = build_indic_indic_rag_fuzzy(
        src_indic=src,
        tgt_indic=tgt,
        sample_per_lang=2000,   # tweak for speed vs quality
        sim_threshold=0.78
    )
    return vs

def identify_intent(state: TranslationState) -> dict:
    """
    Uses Pydantic to extract the source and target language codes and translation query from user's message. Returns a dict containing the extracted fields to update the state
    
    arguments: 
        state: current Langgraph state
    """

    user_message=state["messages"][-1].content

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a language intent extraction agent. Task: analyze user's request, "
         "identify source language, target language and the text that requires translation. "
         "You must output a JSON object that adheres to the schema. "
         "Use ISO 639 codes (eg Hindi='hi', Tamil='ta'). "
         "Assume  source language is the one used in the prompt if we do not have any explicit source mentioned, "
         "and the target language is the one that's explicitly requested by the user or otherwise implied."
        ),
        ("human", "User Request: {user_input}")
    ])

    extraction_chain = prompt | llm.with_structured_output(LanguageIntent) # | is langchain's pipeline operator, passes prompt to LLM with the LanguageIntent schema
    try:
        intent_data = extraction_chain.invoke({
            "user_input": user_message 
        })
        
        return {
            "source_lang": intent_data.source_lang_code,
            "target_lang": intent_data.target_lang_code,
            "query_for_retrieval": intent_data.query_for_retrieval,
        }
    except Exception as e:
        print(f"Error during intent identification: {e}")
        return {
            "source_lang": "error",
            "target_lang": "error",
            "query_for_retrieval": "Error: Could not parse intent",
            "messages": [AIMessage(content=f"could not understand translation request. specify both languages and text clearly. Error: {e}")]
        }
    
def generate_rag_query(state: TranslationState) -> dict:
    """
    just returns query_for_retrieval unchanged. 
    """
    q = state.get("query_for_retrieval", "")
    return {"query_for_retrieval": q}

def retrieve_context(state: TranslationState) -> dict:
    """
    Retrieves the top similar pairs from the Indic-Indic vector DB (built in the third step). Returns a string."""
    src = state.get("source_lang")
    tgt = state.get("target_lang")
    query = state.get("query_for_retrieval")
    if not (src and tgt and query):
        return {"retrieved_context": ""}
    #if store exists in memory
    if (src, tgt) not in _INDIC_VECTOR_STORES:
        _INDIC_VECTOR_STORES[(src, tgt)] = load_or_create_vector_store(
            src, tgt,
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
    store = _INDIC_VECTOR_STORES.get((src, tgt))
    #if not, builds it within the function
    results = store.similarity_search(query, k=5)

    # Format for prompt
    blocks = []
    for r in results:
        md = r.metadata
        blocks.append(
            f"SRC: {md['src_indic']}\n"
            f"TGT: {md['tgt_indic']}\n"
        )

    return {"retrieved_context": "\n".join(blocks).strip()}



def tool_router(state: TranslationState) -> dict:
    src = state.get("source_lang")
    tgt = state.get("target_lang")

    # invalid case
    if not src or not tgt:
        return {"tool_call_result": "direct_translate"}

    # exists
    if (src, tgt) in _INDIC_VECTOR_STORES:
        return {"tool_call_result": "retrieve_context"}

    # try to build
    try:
        _INDIC_VECTOR_STORES[(src, tgt)] = load_or_create_vector_store(
            src, tgt,
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        return {"tool_call_result": "retrieve_context"}

    except Exception as e:
        print("[Router] Could not load/build RAG:", e)
        return {"tool_call_result": "direct_translate"}

    
def direct_translate(state: TranslationState) -> dict:
    """
    Performs final translation.
    Uses the retrieved Indic-Indic dataset to perform RAG translation.
    If context is empty ie dataset not available, then it does plain translation.
    """
    src_lang = state.get("source_lang")
    tgt_lang = state.get("target_lang")
    query = state.get("query_for_retrieval")
    retrieved = state.get("retrieved_context", "").strip()

    system_prompt = f"""You are a highly accurate translation model that translates directly from {src_lang} to {tgt_lang}. 
    Follow these rules strictly:
    - NEVER translate via English.
    - Use ONLY {src_lang} -> {tgt_lang} phrasing.
    - Maintain meaning, tone and politeness.
    - Output ONLY the translated text with no explanation.
    """

    if retrieved:
        context_block = f""" Use the following high-quality {src_lang} -> {tgt_lang} example pairs as context: 
        {retrieved} 
        Now translate the following sentence from {src_lang} to {tgt_lang}:
        {query}
        """
    else:
        # no rag available due to unavailable dataset: does normal translation
        context_block = f"""
        Translate the following sentence from {src_lang} to {tgt_lang}:
        {query}
        """

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.strip()),
        ("human", context_block.strip())
    ])

    response = (final_prompt | llm).invoke({})

    return {"final_translation": response.content}


def build_graph():
    workflow = StateGraph(TranslationState)
    workflow.add_node("identify_intent", identify_intent)
    workflow.add_node("generate_rag_query", generate_rag_query)
    workflow.add_node("tool_router", tool_router)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("direct_translate", direct_translate)

    workflow.add_edge(START, "identify_intent")
    workflow.add_edge("identify_intent", "tool_router")

    # Conditional routing based on the router node's return value
    workflow.add_conditional_edges(
        "tool_router",
        lambda state: state["tool_call_result"],     # Or: return of tool_router
        {
            "retrieve_context": "retrieve_context",
            "direct_translate": "direct_translate",
        }
    )

    #always goes to translation after retrieval
    workflow.add_edge("retrieve_context", "direct_translate")
    workflow.add_edge("direct_translate", END)
    return workflow.compile()

#user input --> langgraph
if __name__ == "__main__":
    print("Indic-Indic RAG Translation ")
    print("Enter a translation request in the language of your choice with instructions on which language to translate to.")
    print("Example: 'मुझे इसे उड़िया में translate करना है: मेरा नाम आकाश है और मुझे वनीला आइसक्रीम बहुत पसंद है।'\n")

    user_text = input("Input: ").strip()

    if not user_text:
        print("No input given, exiting.")
        exit()

    user_msg = HumanMessage(content=user_text)
    initial_state = {"messages": [user_msg]}

    graph = build_graph()
    result = graph.invoke(initial_state)

    print("TRANSLATION RESULTS:")
    print("Detected source language:", result.get("source_lang"))
    print("Detected target language:", result.get("target_lang"))
    print("Text to translate:", result.get("query_for_retrieval"))
    print("\nRetrieved Context (RAG)")
    ctx = result.get("retrieved_context") or ""
    print(ctx if ctx else "[no context retrieved, hence direct translation used]")

    print("\nFINAL TRANSLATION")
    print(result.get("final_translation"))
    print("\n")
