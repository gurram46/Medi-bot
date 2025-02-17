import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env (if present)
load_dotenv(find_dotenv())

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Define the path where the vectorstore is saved (built by memory_.py)
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """
    Load the FAISS vectorstore from disk using the prebuilt embeddings.
    If an error occurs, fallback to a minimal FAISS store.
    """
    try:
        # Initialize the embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Load the vectorstore from the local directory
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading FAISS vectorstore: {e}")
        # Fallback: create a minimal vectorstore from a dummy text
        db = FAISS.from_texts(["This is fallback text."], HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        return db

def set_custom_prompt(custom_prompt_template):
    """
    Return a PromptTemplate instance using the provided template string.
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    """
    Initialize the HuggingFaceEndpoint LLM with explicit parameters.
    """
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",  # Explicitly set the task
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )

def display_source_docs(source_documents):
    """
    Display the source document metadata and snippets in a readable format.
    """
    st.markdown("### Source Documents")
    for i, doc in enumerate(source_documents, start=1):
        st.markdown(f"**Document {i}:**")
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        st.markdown(f"**Source:** {source} | **Page:** {page}")
        snippet = doc.page_content.strip().replace("\n", " ")[:300] + "..."
        st.markdown(f"**Snippet:** {snippet}")
        st.markdown("---")

def main():
    st.title("Ask Chatbot!")

    # Initialize conversation history if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous conversation messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Get new user prompt
    prompt_input = st.chat_input("Pass your prompt here")

    if prompt_input:
        st.chat_message("user").markdown(prompt_input)
        st.session_state.messages.append({"role": "user", "content": prompt_input})

        # Define a custom prompt template (feel free to adjust it)
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know; don't try to make up an answer.
        Dont provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            st.error("Please set your HF_TOKEN environment variable.")
            return

        try:
            # Load vectorstore using the cached function
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return

            # Create the QA chain using the prebuilt vectorstore and LLM
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Invoke the QA chain with the user query
            response = qa_chain.invoke({"query": prompt_input})
            result = response["result"]
            source_documents = response["source_documents"]

            st.markdown("### Answer")
            st.markdown(result)
            display_source_docs(source_documents)

            combined_response = result + "\n\n" + "Source Docs:\n" + "\n".join(
                [f"Doc {i+1}: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})"
                 for i, doc in enumerate(source_documents)]
            )
            st.session_state.messages.append({"role": "assistant", "content": combined_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
