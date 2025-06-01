## 📘 Project: LLM Question-Answering Application 🤖

This project is an interactive **LLM-based question-answering web app** built with **Streamlit**, designed to let users upload their own documents (PDF, DOCX, TXT) and ask questions based on their content.

It leverages **OpenAI's GPT models** to generate answers using context retrieved from the uploaded documents, allowing users to explore and extract knowledge dynamically.

### 🔧 Tools & Technologies

- **OpenAI**: Provides the underlying LLM (e.g., GPT-3.5-Turbo) used for generating context-aware answers.
- **LangChain**: Manages the retrieval and QA chain pipeline, enabling smooth integration between language models and document-based search.
- **FAISS**: Serves as the vector store that indexes document embeddings and supports similarity-based retrieval.
- **Streamlit**: Powers the front-end, making the app simple to use via an intuitive web interface without writing any frontend code.

### 💡 What it does

- Allows users to **upload files** in PDF, DOCX, or TXT format.
- Automatically **splits and embeds** document content into vector representations.
- Retrieves the most relevant chunks using Chroma.
- Passes them to the OpenAI model to generate **natural language answers**.
- Displays results interactively through a web interface.
