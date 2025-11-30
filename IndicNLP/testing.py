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
    
#placeholders for step 6, unused as of now
def generate_rag_query(state: TranslationState) -> dict: return {} 
def retrieve_context(state: TranslationState) -> dict: return {}
def tool_router(state: TranslationState) -> str: return END
def direct_translate(state: TranslationState) -> dict: return {}

def build_graph():
    workflow = StateGraph(TranslationState)
    
    workflow.add_node("identify_intent", identify_intent)
    
    workflow.add_edge(START, "identify_intent")
    #placeholder nodes with no functionality yet, but needed to illustrate the graph process
    workflow.add_node("generate_rag_query", generate_rag_query)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("tool_router", tool_router)
    workflow.add_node("direct_translate", direct_translate)
    #for now, ends after the first step (testing). will update in later steps
    workflow.add_edge("identify_intent", END) 
    return workflow.compile()

from datasets import load_dataset
import pandas as pd
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


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
        ds_src = ds_src.select(random.sample(range(len(ds_src)), sample_per_lang))
    if sample_per_lang < len(ds_tgt):
        ds_tgt = ds_tgt.select(random.sample(range(len(ds_tgt)), sample_per_lang))

    print("Sample sizes:", len(ds_src), len(ds_tgt))

    df_src_raw = pd.DataFrame(ds_src)[["src", "tgt"]]
    df_tgt_raw = pd.DataFrame(ds_tgt)[["src", "tgt"]]

    df_src_raw.columns = ["en_src", "src_indic"]
    df_tgt_raw.columns = ["en_tgt", "tgt_indic"]

    embedder = SentenceTransformerEmbeddings(model_name=embed_model_name)

    en_src_list = df_src_raw["en_src"].tolist()
    en_tgt_list = df_tgt_raw["en_tgt"].tolist()

    en_src_emb = np.array(embedder.embed_documents(en_src_list))
    en_tgt_emb = np.array(embedder.embed_documents(en_tgt_list))

    def normalize(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    en_src_emb = normalize(en_src_emb)
    en_tgt_emb = normalize(en_tgt_emb)

    # nearest neighbor search for english alignment
    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(en_tgt_emb)

    distances, indices = nn.kneighbors(en_src_emb)
    sims = 1 - distances[:, 0]

    print(f"Average similarity: {sims.mean():.4f}")
    print(f"Pairs above threshold {sim_threshold}: {np.sum(sims >= sim_threshold)}")

    # build aligned Indic-Indic rows
    aligned = []
    for i, sim in enumerate(sims):
        if sim >= sim_threshold:
            tgt_idx = indices[i]
            aligned.append({
                "src_indic": df_src_raw.iloc[i]["src_indic"],
                "tgt_indic": df_tgt_raw.iloc[tgt_idx]["tgt_indic"],
                "sim": float(sim),
                "en_src": df_src_raw.iloc[i]["en_src"],
                "en_tgt": df_tgt_raw.iloc[tgt_idx]["en_tgt"]
            })

    aligned_df = pd.DataFrame(aligned)
    print("Aligned Indic–Indic pairs:", aligned_df.shape)

    if aligned_df.empty:
        raise ValueError("No fuzzy matches found")

    metadatas = []
    for _, row in aligned_df.iterrows():
        metadatas.append({
            "src_indic": str(row["src_indic"]),
            "tgt_indic": str(row["tgt_indic"]),
            "sim": float(row["sim"]),
            "en_src": str(row["en_src"]),
            "en_tgt": str(row["en_tgt"]),
        })

    # source indic texts to embed
    texts = [str(x) for x in aligned_df["src_indic"].tolist()]
    ids   = [str(i) for i in range(len(texts))]

    # build vector store
    persist_dir = f"{persist_root}/{src_indic}_{tgt_indic}"
    collection_name = f"indic_fuzzy_{src_indic}_{tgt_indic}"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=SentenceTransformerEmbeddings(model_name=embed_model_name),
        persist_directory=persist_dir
    )

    # add to chroma
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Stored {len(texts)} Indic→Indic aligned pairs at {persist_dir}")

    return vector_store, aligned_df

_INDIC_VECTOR_STORES = {}

vector_store, df_pairs = build_indic_indic_rag_fuzzy(
    "hi", "ta",
    sample_per_lang=800,
    sim_threshold=0.75
)

_INDIC_VECTOR_STORES[("hi", "ta")] = vector_store

