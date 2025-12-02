# Instructions to run
1. Clone the repository
2. If using pip, run `pip install -r requirements.txt` followed by `python3 main.py`
3. If using uv, run `uv sync` followed by `uv run main.py` 
4. To run the user interface, you must have Streamlit installed (part of both `requirements.txt` and `pyproject.toml`). Simply run `streamlit run app.py` 

------

# Overview of MAT496

In this course, we have primarily learned Langgraph. This is helpful tool to build apps which can process unstructured `text`, find information we are looking for, and present the format we choose. Some specific topics we have covered are:

- Prompting
- Structured Output 
- Semantic Search
- Retreaval Augmented Generation (RAG)
- Tool calling LLMs & MCP
- Langgraph: State, Nodes, Graph

We also learned that Langsmith is a nice tool for debugging Langgraph codes.

------

# Capstone Project objective

The first purpose of the capstone project is to give a chance to revise all the major above listed topics. The second purpose of the capstone is to show your creativity. Think about all the problems which you can not have solved earlier, but are not possible to solve with the concepts learned in this course. For example, We can use LLM to analyse all kinds of news: sports news, financial news, political news. Another example, we can use LLMs to build a legal assistant. Pretty much anything which requires lots of reading, can be outsourced to LLMs. Let your imagination run free.


-------------------------

# Project Report
## RAG Agent for Direct NLP Translation between Indic Languages using LangGraph

## Overview

- Most translation systems rely on using English as a pivot language. Especially for Indic languages, the pipeline is such that the source language to translate from first gets translated to English, and that English translation is translated to the target language. 
- This is because high-quality Indic-Indic parallel corpora are very limited whereas there's an abundance of English-Indic datasets. Obviously this has some major drawbacks, mainly a loss of semantic nuance because many Indic languages share the same grammatical patterns, loanwords and even cultural references that do not make any sense in English. When English is the intermediary these nuances are lost.

- **The goal of this project** is a proof-of-concept direct Indic-Indic translation agent. I used Langgraph for my multi-node agent and created a RAG database constructed from fuzzy-aligned Indic language pairs. 
- **Obstacle:** No high-quality Indic-Indic datasets exist. Hence, I used AI4Bharat's Samanantar dataset (https://huggingface.co/datasets/ai4bharat/samanantar) which has English-Indian language pairs. 
- **Creation of Dataset:** 
```
1. I created a fuzzy alignment technique which loads two English-Indic language datasets from the above.
2. I extracted the English sentences from each dataset.
3. I embedded them using mpnet and normalized the embeddings with NumPy.
4. I used scikit-learn's NearestNeighbours learner to find similar English sentences across the two splits.
5. If similarity > threshold, each Indian-Indian translation is a valid aligned pair
6. Stored aligned pairs in Pandas dataframe and embedded source Indic sentences in a ChromaDB database.
```
- The final system removes English entirely from the translation step. The only role of English is during dataset construction, where English sentences act as a semantic placeholder for aligning the two Indic languages. The RAG context fed to the LLM during translation is entirely the two Indic languages only.
- This is to be done for each dataset, which takes a lot of time. However, I have **generated multiple datasets myself** and these can be found in the repository. 
- *Language translations for these pairs will use these datasets and will not take time. For a translation pair not created by me yet, say `Tamil -> Gujarati`, the agent will create this dataset before translating. This will take time (depending on sample size, for a sample size of 4500 it takes 5-8 minutes). Sample size can be scaled as required.*
- To simultaneously create multiple datasets of your liking, run the `create_all_datasets.py` script and modify the language pairs to your liking. 

## Reason for picking up this project

- This project has been a personal interest of mine, and so I figured I may as well implement it with my actual learnings from the course. I also believe that it is a novel idea in that it addresses several resource inefficiences when translating in a highly linguistically diverse region and such a thing would be impossible without the use of agents. If I were to implement it with LangGraph, it'd require pretty much everything that's been covered in the course which is just perfect for me. I also wanted to extend this in the future to use GPU acceleration to use these Indic-Indic datasets to finetune a model instead of relying simply on RAG. But this I believe is a very good first step towards that goal.

------
## Video Summary Link: 

https://www.youtube.com/watch?v=TFbyrq88AfI

------

## Plan

I plan to excecute these steps to complete my project.

- [DONE] Step 1: Setting up Python venv, installing all required libraries (langchain, langgraph, pydantic, chromadb maybe?), defining the translation state schema and initializing the LLM (I may either use IndicTrans or gpt-4o-mini) 
- [DONE] Step 2: Define Pydantic model for identifying language, create identification node to use prompting to parse user input into source_lang, target_lang and query_for_retrieval fields in the translation state schema. EDIT: Also added mock nodes/functions (retrieve_context, generate_rag_query, etc) to illustrate workflow.

- [DONE] Step 3: Create RAG database for semantic search, create multilingual embedding model, use it to embed the sentences and store them in the vector store. 

- Addendum: 
  This step was a lot more complicated than I thought. Used an external dataset (Samanantar) from HuggingFace which has a parallel corpus of Indic languages paired with English. However the entire goal of the project is to skip using English as a pivot language. So I fuzzy matched the English sentences across two Samanantar dataset splits as aligned pairs. 
  eg: en->hindi has an English sentence: Police started investigation.
  en->oriya has an English sentence: Police have begun investigating the case.
  These have nearly similar meaning so we treat the Hindi and Oriya translations as an aligned pair. This will be the basis for RAG.

- [DONE] Step 4: Define a direct translation tool and tool router node which would be a tool call to decide whether to use RAG or proceed directly to translation
  Extras: Added a testing.py integrated with uv to directly execute code. Will be updating this with the finished product when I am done with the final step in the notebook. 
  Added the context retrieval node from step 5 here itself.
- [DONE] Step 5: Implement the direct translation node and context retrieval node.
- [DONE] Step 6: Build the final graph, connecting the nodes and defining conditional edges. 
- [DONE] Step 7: Make translation node use the RAG's retrieved context in the final generated translation
Note: It turns out I've already finished step 7 as I can use the direct translation tool as the final translation node and I have implemented the same in Step 6. 
Extras: I also wrote a test call to see if the graph works at every step. Output can be seen in the notebook at this step. It works as intended!
- [DONE] Step 8: Testing step. I have ran the code as a Python script and the output is feasible. It has been added to the repo, and instructions on running it with uv will be given.

## Conclusion:

The project succeeded in building a functional NLP Indic translation agent. It uses my own RAG datasets for contextual accuracy and entirely avoids English during the translation stage. The only use of English is during dataset alignment which is necessary as there is no Indic-Indic corpora to train with.
**Limitations**: 
- I wanted to be able to generate the datasets much quicker. However, each language pair takes up to 5-8 minutes to generate even with a sample size of just 4500 sentences per language. This is not feasible for large-scale use and I need to figure out a way, like GPU acceleration, to generate the datasets faster. 
- The alignment is imperfect due to it being fuzzy. This again is due to the limited dataset size (if I increase sample size, it will take multiple hours per dataset). 
- Despite these problems I've been able to demonstrate the feasibility of direct Indic-Indic translation. If you use one of my pre-generated datasets, you'll be able to run the translation quite quickly and accurately. 

----------

# Added instructions:

- This is a `solo assignment`. Each of you will work alone. You are free to talk, discuss with chatgpt, but you are responsible for what you submit. Some students may be called for viva. You should be able to each and every line of work submitted by you.

- `commit` History maintenance.
  - Fork this respository and build on top of that.
  - For every step in your plan, there has to be a commit.
  - Change [TODO] to [DONE] in the plan, before you commit after that step. 
  - The commit history should show decent amount of work spread into minimum two dates. 
  - **All the commits done in one day will be rejected**. Even if you are capable of doing the whole thing in one day, refine it in two days.  
 
 - Deadline: Nov 30, Sunday 11:59 pm


# Grading: total 25 marks

- Coverage of most of topics in this class: 20
- Creativity: 5
  