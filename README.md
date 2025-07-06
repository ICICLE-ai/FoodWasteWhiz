# Food Waste Ontology Chatbot

### **Overview**
This project provides an interactive chatbot interface to explore and build structured ontologies for food waste. It uses **Meta-Llama-3.1-8B-Instruct** with **Retrieval-Augmented Generation (RAG)** to answer questions based on uploaded PDFs. Users can provide feedback on the model’s answers, edit them, and store evaluations in a local SQLite database.

**Tags**: Smart-Foodsheds

---
## **Acknowledgements**

- National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)

---

#### **Project Structure**

```
.
├── data/                  # Input PDFs
├── processed_data/        # Text extracted from PDFs
├── db/                    # Chroma vector DB
├── feedback.db            # SQLite feedback database
├── hf_token.txt           # Your Hugging Face token
├── main.py                # Main Streamlit app
└── README.md              # This file
```

---

## **How to Guide**

#### **1. Install Dependencies**

Use Python 3.10+. Install dependencies with:

```bash
pip install -r requirements.txt
```

#### **2. HuggingFace Token**

Create a file called `hf_token.txt` in the root directory with your Hugging Face token:

```
your_huggingface_token_here
```

---

#### **3. Place PDF Files**

Put all input PDF files into the `./data/` directory.

---

#### **4. Run the App**

Run the chatbot locally:

```bash
streamlit run app.py
```

---
## **Explanation**

- PDF-to-Text Extraction  
- Embedding + Vector Database (ChromaDB + Sentence Transformers)  
- RAG-based Q&A powered by HuggingFace Transformers  
- Feedback system with editable responses and star rating  
- SQLite backend for storing evaluations  
- Streamlit frontend  

---
### **Usage Flow**

1. Select a predefined question or enter a custom one.
2. The chatbot fetches relevant chunks from uploaded documents.
3. The LLM generates a response using those chunks.
4. You can view, edit, rate, and submit feedback.
5. All feedback is stored in `feedback.db`.

---

### **Future Improvements**

- Authentication and user tracking
- Cloud deployment
- Analytics dashboard for feedback insights
- More advanced reward modeling (embedding similarity, BLEU, etc.)
- Reinforcement Learning with PPO for enhanced knowledge

---

