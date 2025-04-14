# **Conversational Customer Support Agent (RAG)**

Welcome to the **Conversational Customer Support Agent** project! This repository contains the code for a customer support chatbot built using **Retrieval-Augmented Generation (RAG)**. The chatbot leverages **FAISS vector search**, **OpenAI embeddings**, and **LangChain conversational chains** to provide accurate, context-aware responses to user queries based on product documentation.

---

## **Features**

- **Retrieval-Augmented Generation (RAG)**:
  Combines retrieval of relevant documents with generative AI for effective query resolution.
  
- **FAISS Vector Search**:
  Efficient similarity search over product documentation and FAQs.

- **OpenAI Embeddings**:
  Uses OpenAI's `text-embedding-ada-002` model to generate vector embeddings.

- **Multi-turn Memory**:
  Maintains conversation history for coherent multi-turn interactions.

- **Escalation Logic**:
  Automatically escalates complex queries to human support agents when necessary.

- **Modular Design**:
  Easily extendable and customizable for different use cases.

---

## **Installation**

### **1. Clone the Repository**
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

### **2. Set Up a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```


### **3. Install Dependencies**
Install all required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```


### **4. Set Up OpenAI API Key**
Since you are using OpenAI embeddings, ensure your API key is set in your `.bash_profile` or equivalent shell configuration file.

Add the following line to your `.bash_profile` (or `.zshrc`, `.bashrc`, etc.):

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```
Then reload your shell configuration:

```bash
source ~/.bash_profile # Or source ~/.zshrc, depending on your shell
```