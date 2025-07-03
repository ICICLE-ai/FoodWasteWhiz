# Food Waste Ontology Chatbot

### **Overview**
This project provides an interactive chatbot interface to explore and build structured ontologies for food waste. It uses **Meta-Llama-3.1-8B-Instruct** with **Retrieval-Augmented Generation (RAG)** to answer questions based on uploaded PDFs. Users can provide feedback on the model’s answers, edit them, and store evaluations in a local SQLite database. You can also fine-tune the model using reinforcement learning (PPO) on the feedback.

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
├── train_rl.py            # PPO fine-tuning script
└── README.md              # This file
```

---

## **How to Guide**

#### **1. Install Dependencies**

Use Python 3.10+. Install dependencies with:

```bash
pip install -r requirements.txt
```

Make sure `requirements.txt` includes:
```txt
streamlit
pdfplumber
sentence-transformers
langchain
transformers
torch
trl
langchain-chroma
langchain_huggingface
langchain_community
bitsandbytes
trl
transformers
llama_index
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
- PPO-based fine-tuning using TRL for human feedback learning  
- Streamlit frontend  

---
### **Usage Flow**

1. Select a predefined question or enter a custom one.
2. The chatbot fetches relevant chunks from uploaded documents.
3. The LLM generates a response using those chunks.
4. You can view, edit, rate, and submit feedback.
5. All feedback is stored in `feedback.db`.

---

### **Reinforcement Learning with PPO (Optional)**

You can fine-tune the language model using PPO (Proximal Policy Optimization) with the human feedback stored in the database.

### **Steps:**

1. **Prepare Feedback Data:**
   - Ensure `feedback.db` exists and has meaningful `question`, `edited_response`, and `score` entries.

2. **Run the PPO Script:**
   ```bash
   python train_rl.py
   ```

3. **What It Does:**
   - Loads data from `feedback.db`
   - Initializes `Llama-3.1-8B-Instruct` with value head
   - Computes a reward based on token overlap and score
   - Applies PPO updates to align the model closer to human-edited responses
   - Saves the fine-tuned model in `./rl_model/`

4. **Next Steps:**
   - Replace the original model in `main.py` with your fine-tuned version if desired.

---

### **Future Improvements**

- Authentication and user tracking
- Cloud deployment
- Analytics dashboard for feedback insights
- More advanced reward modeling (embedding similarity, BLEU, etc.)

---

