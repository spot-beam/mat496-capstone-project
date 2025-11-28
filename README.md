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

# Project report Template

## Title: RAG for Direct NLP Translation between Indic Languages using LangGraph

## Overview

- In the status quo today, the standard for translating between language pairs (especially Indic languages which don't have sophisticated corpus) is to use English as a pivot language, ie, the source language gets translated to English and the translated English text gets translated to the target language. Obviously this degrades translation quality as we lose a lot of context when translating twice. 
- With Indic languages especially, plenty of similarities arise between languages from grammatical syntax to subject verb agreement to loanwords and highly similar words. 
- My overarching goal is to eventually finetune a pre-existing LLM to exploit these linguistic similarities in aiding direct translation, but this is the one limitation of LangGraph in that I cannot use LangGraph to finetune a model. However I've come to figure out that RAG is a very feasible substitute and a much simpler one at that, so both in the context of this course and for my future prospects in working with LLMs, it's the best first step. We'll be retrieving validated examples of translated sentences from a vector database and using them in the LLM's prompts so our translation will be highly context aware.  

## Reason for picking up this project

- This project has been a personal interest of mine, and so I figured I may as well implement it with my actual learnings from the course. I also believe that it is a novel idea in that it addresses several resource inefficiences when translating in a highly linguistically diverse region and such a thing would be impossible without the use of agents. If I were to implement it with LangGraph, it'd require pretty much everything that's been covered in the course which is just perfect for me. 
To be more specific:
- The core of the application is a large conditional graph with multiple nodes that requires a RAG vector store database that utilizes embeddings which is then queried using semantic search. We'll also be using tools to determine when to use this translation functionality. 
- We'll be using a pydantic schema to extract source and target languages along with the query which  falls under structured outputs and prompting.

## Plan

I plan to excecute these steps to complete my project.

- [DONE] Step 1: Setting up Python venv, installing all required libraries (langchain, langgraph, pydantic, chromadb maybe?), defining the translation state schema and initializing the LLM (I may either use IndicTrans or gpt-4o-mini) 
- [DONE] Step 2: Define Pydantic model for identifying language, create identification node to use prompting to parse user input into source_lang, target_lang and query_for_retrieval fields in the translation state schema. EDIT: Also added mock nodes/functions (retrieve_context, generate_rag_query, etc) to illustrate workflow.

- [DONE] Step 3: Create RAG database for semantic search, create multilingual embedding model, use it to embed the sentences and store them in the vector store. 

### ADDENDUM: 
This step was a lot more complicated than I thought. Used an external dataset (Samanantar) from HuggingFace which has a parallel corpus of Indic languages paired with English. However the entire goal of the project is to skip using English as a pivot language. So I fuzzy matched the English sentences across two Samanantar dataset splits as aligned pairs. 
eg: en->hindi has an English sentence: Police started investigation.
en->oriya has an English sentence: Police have begun investigating the case.
These have nearly similar meaning so we treat the Hindi and Oriya translations as an aligned pair. This will be the basis for RAG.
---

- [TODO] Step 4: Define a direct translation tool and tool router node which would be a tool call to decide whether to use RAG or proceed directly to translation
- [TODO] Step 5: Implement the nodes that require RAG (unnamed as of yet) and a context retrieval node for semantic search 
- [TODO] Step 6: Build the final graph, connecting the nodes and defining conditional edges.
- [TODO] Step 7: Make translation node use the RAG's retrieved context in the final generated translation
- [TODO] Step 8: Testing step. Have to think of what to do and whether any other steps are required once I finish the above steps.

## Conclusion:

STILL A WORK IN PROGRESS 
I had planned to achieve {this this}. I think I have/have-not achieved the conclusion satisfactorily. The reason for your satisfaction/unsatisfaction.

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
  