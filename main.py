#pip install -q transformers peft  accelerate bitsandbytes safetensors sentencepiece streamlit chromadb langchain sentence-transformers gradio pypdf

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

import os
import gradio as gr

import chromadb
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

model_name = "mistralai/Mistral-7B-v0.1"

def load_quantized_model(model_name:str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    return model
def initialize_tokenizer(model_name:str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1
    return tokenizer 
model = load_quantized_model(model_name)
tokenizer = initialize_tokenizer(model_name)
stop_token_ids = [0]
db = "./catalog"

loader = PyPDFDirectoryLoader(db)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(documents)
embed_name = "sentence-transformer/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(embed_name, model_kwargs=model_kwargs)

vectordb = Chroma.from_documents(all_splits, embeddings, persist_directory="chroma_db")

retriever = vectordb.as_retriever()

# build huggingface pipeline for using zephyr-7b-alpha
pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2048,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

# specify the llm
llm = HuggingFacePipeline(pipeline=pipeline)

# build conversational retrieval chain with memory (rag) using langchain
def create_conversation(query: str, chat_history: list) -> tuple:
    try:

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=False
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h,
        )

        result = qa_chain({'question': query, 'chat_history': chat_history})
        chat_history.append((query, result['answer']))
        return '', chat_history


    except Exception as e:
        chat_history.append((query, e))
        return '', chat_history
with gr.Blocks() as demo:
    
    chatbot = gr.Chatbot(label='Chat with your data (Zephyr 7B Alpha)')
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(create_conversation, [msg, chatbot], [msg, chatbot])

demo.launch()