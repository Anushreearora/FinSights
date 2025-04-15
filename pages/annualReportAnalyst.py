import streamlit as st
import os
import tempfile
import fitz
import io
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from streamlit_pdf_viewer import pdf_viewer

from apikey import OPENAI_API_KEY  # Store your OpenAI key in apikey.py

# ---------- SETUP ----------
st.set_page_config(page_title="Annual Report Analyzer")
st.markdown("<h1 style='text-align: center;'>Annual Report Analyzer</h1>", unsafe_allow_html=True)
st.subheader("Upload a company annual/integrated report to explore its contents.")

# ---------- CUSTOM PROMPT ----------
CUSTOM_PROMPT_TEMPLATE = """
Use the following context from a company's annual report to answer the question. 
Be specific, avoid making up information, and cite exact phrases.

{context}

Question: {question}

Answer in valid JSON format:
{{
  "answer": "Detailed answer",
  "sources": "Relevant excerpts from the text"
}}
"""

CUSTOM_PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ---------- HELPER FUNCTIONS ----------

def extract_documents_from_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
    loader = PyPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)
    return docs

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

@st.cache_resource
def initialize_model():
    return ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

@st.cache_resource
def setup_qa(_documents):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(_documents)
    vector_store = FAISS.from_documents(chunks, embedding=get_embeddings())
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=initialize_model(),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT},
    )

def locate_pages_with_terms(doc, excerpts):
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        if any(page.search_for(text) for text in excerpts):
            pages.append(i)
    return pages if pages else [0]

def generate_highlight_annotations(doc, excerpts):
    annotations = []
    for page_num, page in enumerate(doc):
        for excerpt in excerpts:
            for inst in page.search_for(excerpt):
                annotations.append({
                    "page": page_num + 1,
                    "x": inst.x0,
                    "y": inst.y0,
                    "width": inst.x1 - inst.x0,
                    "height": inst.y1 - inst.y0,
                    "color": "red",
                })
    return annotations

# ---------- UI LOGIC ----------

uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.read()
    with st.spinner("Processing document..."):
        docs = extract_documents_from_file(io.BytesIO(file_bytes))
        st.session_state.doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")

    if docs:
        with st.spinner("Initializing retrieval system..."):
            qa = setup_qa(docs)
            st.success("Ready to analyze your report!")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Hi! Ask me anything about the report."}
            ]

        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        if query := st.chat_input("Ask a question about this report..."):
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            with st.spinner("Thinking..."):
                try:
                    result = qa.invoke({"query": query})
                    parsed = json.loads(result["result"])
                    answer = parsed["answer"]
                    sources = parsed["sources"]
                    excerpts = sources.split(". ") if sources else []

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.chat_message("assistant").write(answer)

                    st.session_state.excerpts = excerpts
                    st.session_state.chat_occurred = True

                except Exception as e:
                    st.error(f"Failed to parse response: {str(e)}")

        # PDF Viewer with highlights
        if st.session_state.get("chat_occurred"):
            doc = st.session_state.doc
            excerpts = st.session_state.get("excerpts", [])
            pages_with_highlights = locate_pages_with_terms(doc, excerpts)
            annotations = generate_highlight_annotations(doc, excerpts)

            st.markdown("### Report Viewer with Highlights")
            col1, col2, col3 = st.columns([1, 3, 1])
            if "current_page" not in st.session_state:
                st.session_state.current_page = pages_with_highlights[0]

            with col1:
                if st.button("Previous") and st.session_state.current_page > 0:
                    st.session_state.current_page -= 1
            with col2:
                st.write(f"Page {st.session_state.current_page + 1}")
            with col3:
                if st.button("Next") and st.session_state.current_page < len(doc) - 1:
                    st.session_state.current_page += 1

            pdf_viewer(
                file_bytes,
                width=700,
                height=800,
                annotations=annotations,
                pages_to_render=[st.session_state.current_page],
            )
