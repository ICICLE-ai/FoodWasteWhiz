import os
import time
import warnings
import logging
import sqlite3

import streamlit as st
warnings.filterwarnings('ignore')

import pdfplumber
from llama_index.core import SimpleDirectoryReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import HumanMessage, AIMessage

import torch
import transformers

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

###############################################################################
# LOGGING CONFIGURATION
###############################################################################
# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

###############################################################################
# 1) CONFIG & SECURITY
###############################################################################
class Config:
    """Reads HuggingFace token from file. Adjust as needed."""
    def __init__(self, hf_token_file="hf_token.txt"):
        if not os.path.exists(hf_token_file):
            raise FileNotFoundError(
                f"HF token file '{hf_token_file}' not found. "
                "Please create it and place your token inside."
            )
        with open(hf_token_file, "r") as f:
            self.hf_token = f.read().strip()

###############################################################################
# 2) DOCUMENT PROCESSOR
###############################################################################
class DocumentProcessor:
    """
    Handles PDF to text conversion and building the vector store from text files.
    """
    def __init__(self, input_dir, output_dir, persist_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.persist_dir = persist_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def process_pdfs(self):
        """
        Loop through input_dir, convert PDF to text in output_dir.
        (If you only want to do it once, move this logic to a separate script
         or gate it behind a check for existing .txt files.)
        """
        logger.info("Starting PDF processing...")
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(self.input_dir, filename)
                self._extract_text_from_pdf(file_path)

    def _extract_text_from_pdf(self, file_path):
        output_file = os.path.join(
            self.output_dir,
            os.path.basename(file_path).replace(".pdf", ".txt")
        )
        if os.path.exists(output_file):
            logger.info(f"Already processed: {output_file}")
            return

        with pdfplumber.open(file_path) as pdf, open(output_file, "w", encoding="utf-8") as out_f:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    out_f.write(text + "\n")
        logger.info(f"Processed PDF -> {output_file}")

    def build_vectorstore(self):
        """
        Builds (or loads) a Chroma-based vector store from text in output_dir.
        """
        logger.info("Building/Loading vector store...")

        reader = SimpleDirectoryReader(input_dir=self.output_dir)
        documents = reader.load_data()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        formatted_docs = [
            Document(page_content=doc.text, metadata=doc.metadata)
            for doc in documents
        ]
        all_splits = splitter.split_documents(formatted_docs)

        # Embeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )

        # If persist_dir has an existing DB, load it. Else create from docs.
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            logger.info("Found existing DB, loading it.")
            vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings
            )
        else:
            logger.info("No existing DB found, creating new one.")
            os.makedirs(self.persist_dir, exist_ok=True)
            vectordb = Chroma.from_documents(
                documents=all_splits,
                embedding=embeddings,
                persist_directory=self.persist_dir
            )

        return vectordb

###############################################################################
# 3) SQLite
###############################################################################
def init_db():
    logger.info("Initializing SQLite database...")
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    # Adjusted schema: add `edited_response`
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            original_response TEXT,
            edited_response TEXT,
            time REAL,
            score REAL
        )
    ''')
    conn.commit()
    conn.close()

def store_feedback(question, original_response, edited_response, time, score):
    logger.info("Storing feedback in the database...")
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO feedback (question, original_response, edited_response, time, score)
        VALUES (?, ?, ?, ?, ?)
        """,
        (question, original_response, edited_response, time, score)
    )
    conn.commit()
    conn.close()

def get_feedback():
    logger.info("Fetching feedback from database...")
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute("SELECT * FROM feedback ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

###############################################################################
# 4) LLM MANAGER
###############################################################################
@st.cache_resource
def load_llm_pipeline(model_id: str, hf_auth: str):
    """
    Loads the model & tokenizer once and returns a pipeline.
    Uses @st.cache_resource so we don't reload on each re-run.
    """
    logger.info(f"Loading model config for {model_id}...")
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32
    )

    logger.info("Loading model & tokenizer...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=hf_auth
    )
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, use_auth_token=hf_auth
    )

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        temperature=0.7,
        top_p=0.7,
        max_new_tokens=4096,
        repetition_penalty=1.00
    )

    # Wrap in a HuggingFacePipeline for usage in langchain
    llm = HuggingFacePipeline(pipeline=generate_text)
    logger.info("LLM pipeline loaded successfully.")
    return llm

def create_rag_chain(llm, vectordb: Chroma):
    """
    Creates the retrieval-augmented generation chain using the given LLM and vectordb.
    """
    system_prompt = """
        You are a food waste expert. Your goal is to develop a comprehensive and structured ontology using your knowledge and retrieved context when asked to build an ontology.

        ### Response Guidelines for creating ontology:
        1. Ontology Construction
        - Define **classes, subclasses, properties, and relationships** clearly.
        - Organize concepts into **hierarchies, taxonomies, and classification systems** with logical consistency.

        2. Context Integration & Reasoning
        - Use retrieved context to refine answers only when it improves the result; highlight any missing information.
        - Apply ontology principles (e.g., subsumption, part-whole relationships) for accuracy.

        3. Clear & Structured Explanations
        - Use step-by-step reasoning, and precise definitions.
        - Summarize insights while ensuring completeness.

        4. Iterative Refinement
        - Improve ontology with new context and suggest refinements.
        - Seek clarification when necessary before proceeding.

        For all other questions, provide a simple, concise, and clear answer without unnecessary details.

        Retrieved Context: {context}

        {history}
        Human: {input}
        Assistant:
    """
    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["history", "input", "context"]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectordb.as_retriever()

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    logger.info("Created retrieval-augmented generation chain.")
    return rag_chain

###############################################################################
# 5) STREAMLIT CACHED FUNCTIONS TO PROCESS DOCS / LOAD VECTORSTORE
###############################################################################
@st.cache_data
def process_pdfs_once(input_dir, output_dir):
    """
    If you still want Streamlit to do the PDF -> text conversion, 
    wrap it in a cached function so it runs only once unless the code changes.
    """
    doc_proc = DocumentProcessor(input_dir, output_dir, "")
    doc_proc.process_pdfs()
    return True

@st.cache_resource
def build_or_load_vectorstore_cached(output_dir, persist_dir):
    """
    Builds or loads the Chroma DB. Runs only once unless code changes.
    """
    doc_proc = DocumentProcessor("", output_dir, persist_dir)
    vectordb = doc_proc.build_vectorstore()
    return vectordb

###############################################################################
# 6) THE STREAMLIT APP
###############################################################################
def main():
    st.title("Food Waste Ontology Chatbot")
    logger.info("App started. Rendering UI...")

    # 1) Load config / HF token
    try:
        config = Config("hf_token.txt")
        hf_auth = config.hf_token
        logger.info("HuggingFace token loaded.")
    except FileNotFoundError as e:
        st.error(str(e))
        logger.error(str(e))
        return

    # 2) Setup directories
    input_dir = "./data"
    output_dir = "./processed_data"
    persist_dir = "./db"

    # 3) (Optional) Run the PDF processing once
    #    If you want to do this outside Streamlit, remove or comment out
    if process_pdfs_once(input_dir, output_dir):
        logger.info("PDF processing completed or skipped if already done.")

    # 4) Build or load the vectorstore
    vectordb = build_or_load_vectorstore_cached(output_dir, persist_dir)

    # 5) Load LLM pipeline
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    logger.info(f"Using model -> {model_id}")
    llm = load_llm_pipeline(model_id, hf_auth)

    # 6) Build retrieval chain
    if "rag_chain" not in st.session_state:
        st.session_state["rag_chain"] = create_rag_chain(llm, vectordb)

    # 7) Predefined questions or custom question
    questions = [
        "What are the main causes of food waste?",
        "How can tomato waste be valorized?",
        "What are the environmental impacts of waste?",
        "How can we reduce aesthetic-related waste in bananas?",
        "What are the regulations for banana exports to the EU?",
        "Other (specify below)"
    ]

    selected_question = st.selectbox(
        label="Select a question or choose 'Other' to specify your own:",
        options=questions
    )

    custom_question = ""
    if selected_question == "Other (specify below)":
        custom_question = st.text_input("Please specify your question:")

    # Determine the final question to use
    user_input = custom_question if custom_question.strip() != "" else selected_question
    st.write(f"**Selected Question:** {user_input}")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # We store the current question/answer in session state so feedback
    # can be submitted before it goes into the conversation history.
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = None
    if "current_answer" not in st.session_state:
        st.session_state["current_answer"] = None
    if "response_time" not in st.session_state:
        st.session_state["response_time"] = None

    # -- SEND button logic
    if st.button("Send") and user_input.strip():
        logger.info(f"User clicked 'Send' with question: {user_input}")
        start_time = time.time()

        # Run the RAG chain
        rag_chain = st.session_state["rag_chain"]
        retriever = vectordb.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(user_input, k=6)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."

        # Format existing conversation history as text
        history_str = format_history(st.session_state["history"])

        chain_inputs = {
            "history": history_str,
            "input": user_input,
            "context": context
        }
        result = rag_chain.invoke(chain_inputs)
        logger.info(f"Raw Result: {result}")
        assistant_response = result["answer"].split("Assistant:")[-1].strip()
        logger.info(f"Cropped Result: {assistant_response}")

        response_time = round(time.time() - start_time, 2)
        logger.info("LLM response received.")

        # We do NOT immediately push question/answer to history.
        # Instead, store them in session state until feedback is submitted.
        st.session_state["current_question"] = user_input
        st.session_state["current_answer"] = assistant_response
        st.session_state["response_time"] = response_time

    if st.session_state["current_question"] and st.session_state["current_answer"]:
        st.markdown("###### Assistant Response")
        st.write("**Original Answer:**")
        st.write(st.session_state["current_answer"])
        
        # Provide a text area to let the user edit the answer:
        edited_answer = st.text_area(
            "Edit the answer as needed:",
            value=st.session_state["current_answer"],  # Default to the original
            key="edited_answer"
        )

        st.write(f"**Response Time:** {st.session_state['response_time']} seconds")

        score = st.slider(
            "Rate the response (0 - Poor, 1 - Excellent)",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="feedback_slider"
        )

        if st.button("Submit Feedback"):
            logger.info("User clicked 'Submit Feedback' button.")
            # Save feedback into DB including edited response
            store_feedback(
                st.session_state["current_question"],
                st.session_state["current_answer"],      # original
                edited_answer,                            # edited
                st.session_state["response_time"],
                score
            )
            st.success("Feedback recorded! Thank you.")

            # Now move the Q&A into conversation history
            st.session_state["history"].append(HumanMessage(content=st.session_state["current_question"]))
            st.session_state["history"].append(AIMessage(content=edited_answer))

            # Reset the placeholders
            st.session_state["current_question"] = None
            st.session_state["current_answer"] = None
            st.session_state["response_time"] = None

    # Display conversation history
    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("###### Conversation History")
        for msg in st.session_state["history"]:
            if isinstance(msg, HumanMessage):
                st.markdown(f"**Human**: {msg.content}")
            else:
                st.markdown(f"**Assistant**: {msg.content}")

def format_history(history):
    """
    Utility to produce the system's string from conversation history.
    """
    history_str = ""
    for message in history:
        if isinstance(message, HumanMessage):
            history_str += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            history_str += f"Assistant: {message.content}\n"
    return history_str

if __name__ == "__main__":
    main()
