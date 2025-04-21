import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)
print("Loaded API Key:", os.getenv("OPENAI_API_KEY"))

# Set page configuration
st.set_page_config(
    page_title="BookyBot: News Research Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main title and description
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .description {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    <div class="main-title">BookyBot: News Research Tool 📈</div>
    <div class="description">Analyze and summarize news articles with AI-powered insights.</div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for URL input
st.sidebar.header("🔗 Enter News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", placeholder="Enter a valid news article URL")
    urls.append(url)

process_url_clicked = st.sidebar.button("🚀 Process URLs")
file_path = "faiss_store_openai.pkl"

# Main content layout
main_placeholder = st.empty()
query_placeholder = st.empty()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)

if process_url_clicked:
    with st.spinner("🔄 Processing URLs... Please wait."):
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.markdown("### Step 1: Loading Data... ✅")
        data = loader.load()
        if not data:
            st.error("❌ No data could be loaded from the provided URLs. Please check the URLs.")
            print("Data loading failed. Check the URLs.")
        else:
            print(f"Loaded data: {data}")

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.markdown("### Step 2: Splitting Text... ✅")
        docs = text_splitter.split_documents(data)
        if not docs:
            st.error("❌ No documents could be created from the loaded data.")
            print("Text splitting failed. Check the input data format.")
        else:
            print(f"Split documents: {docs}")

        # Create embeddings and save to FAISS index
        if docs:
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.markdown("### Step 3: Building Embedding Vectors... ✅")
            time.sleep(2)

            # Save the FAISS index to a pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

# Query input and results
st.markdown("---")
st.markdown("## 🔍 Ask a Question About the Articles")
query = query_placeholder.text_input("Type your question here:", placeholder="e.g., What is the main topic of the articles?")
if query:
    if os.path.exists(file_path):
        with st.spinner("🔄 Retrieving answer..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                # result will be a dictionary of this format --> {"answer": "", "sources": [] }
                st.markdown("### ✅ Answer")
                st.success(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.markdown("### 📚 Sources")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(f"- {source}")




