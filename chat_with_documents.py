# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt
import streamlit as st
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
  import os
  name, extension = os.path.splitext(file)

  if extension == '.pdf':
    from langchain_community.document_loaders import PyPDFLoader
    print(f'Loading {file}')
    loader = PyPDFLoader(file)
  elif extension == '.docx':
    from langchain_community.document_loaders import Docx2txtLoader
    print(f'Loading {file}')
    loader = Docx2txtLoader(file)
  elif extension == '.txt':
    from langchain_community.document_loaders import TextLoader
    print(f'Loading {file}')
    loader = TextLoader(file)
  else:
    print(f'Unsupported file type: {extension}')
    return None
  
  data = loader.load()
  return data

# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  chunks = text_splitter.split_documents(data)
  return chunks

# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
  embeddings = OpenAIEmbeddings() # 512 works as well
  vector_store = Chroma.from_documents(chunks, embeddings)
  
  # if you want to use a specific directory for chromadb
  # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
  return vector_store

def ask_and_get_answer(vector_store, q, k=3):
  from langchain.chains import RetrievalQA
  from langchain_openai import ChatOpenAI

  llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)

  retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": k})
  chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=retriever
  )

  answer = chain.invoke(q)
  return answer['result']

# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
  import tiktoken
  enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
  total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
  #print(f"Total tokens: {total_tokens}")
  #print(f"Embedding Cost in USD: ${total_tokens / 1000 * 0.0004:.6f}")
  return total_tokens, total_tokens / 1000 * 0.0004

# clear the chat history from streamlit session state
def clear_history():
  if 'history' in st.session_state:
    del st.session_state['history']

#if __name__ == "__main__":
  #import os

# loading the OpenAI api key from .env
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv(), override=True)

st.header('LLM Question-Answering Application 🤖')

st.image('img.png')
with st.sidebar:
  # text_input for the OpenAI API key (alternative to python-dotenv and .env)
  api_key = st.text_input('Open API key:', type='password')
  if api_key:
    os.environ['OPENAI_API_KEY'] = api_key

    # file uploader widget
    uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

    # chunk size number widget
    chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

    # k number input widget
    k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

    # add data button widget
    add_data = st.button('Add data', on_click=clear_history)

    if uploaded_file and add_data: # if the user browsed a file
      with st.spinner('Reading, chunking and embedding file ...'):

        # writing the file from RAM to the current directory on disk
        bytes_data = uploaded_file.read()
        file_name = os.path.join('./', uploaded_file.name)
        with open(file_name, 'wb') as f:
          f.write(bytes_data)

        data = load_document(file_name)
        chunks = chunk_data(data, chunk_size=chunk_size)
        st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

        tokens, embedding_cost = calculate_embedding_cost(chunks)
        st.write(f'Embedding cost: ${embedding_cost:.4f}')

        # creating the embeddings and returning the Chroma vector store
        vector_store = create_embeddings(chunks)

        # saving the vector store in the streamlit session state (to be persistent between reruns)
        st.session_state.vs = vector_store
        st.success('File uploaded, chunked and embedded successfully.')

  else:
    st.warning("Please enter your OpenAI API Key to continue.")

# user's question text input widget
q = st.text_input('Ask a question about the content of your file:')
if q: # if the user entered a question and hit enter
  if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
    vector_store = st.session_state.vs
    answer = ask_and_get_answer(vector_store, q, k)
    
    st.markdown("""
        <style>
        .chat-bubble {
            max-width: 80%;
            padding: 10px 15px;
            margin: 8px;
            border-radius: 15px;
            font-size: 15px;
            line-height: 1.4;
            word-wrap: break-word;
        }
        .user {
            background-color: #005c4b;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 0;
            margin-left: auto;
            margin-right: 0;
        }
        .bot {
            background-color: #262d31;
            color: white;
            align-self: flex-start;
            border-bottom-left-radius: 0;
            margin-right: auto;
            margin-left: 0;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-top: 20px;
            margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Mostrar respuesta actual
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble bot">A: {answer}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Historial en sesión
    if 'history' not in st.session_state:
      st.session_state.history = ''

    # Guardar la pregunta y respuesta actuales
    value = f'Q: {q} \nA: {answer}'
    st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
   
  # Mostrar historial como burbujas
  h = st.session_state.history
  historial_items = h.strip().split('-' * 100)
  st.markdown('<div class="chat-container">', unsafe_allow_html=True)
  for item in reversed(historial_items):
    lines = item.strip().split('\n')
    q_line = next((line for line in lines if line.startswith('Q:')), None)
    a_line = next((line for line in lines if line.startswith('A:')), None)
    if q_line and a_line:
      st.markdown(f'<div class="chat-bubble user">{q_line.strip()}</div>', unsafe_allow_html=True)
      st.markdown(f'<div class="chat-bubble bot">{a_line.strip()}</div>', unsafe_allow_html=True)
  st.markdown('</div>', unsafe_allow_html=True)
# run the app: streamlit run ./chat_with_documents.py