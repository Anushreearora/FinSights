import streamlit as st
import requests
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from apikey import DCF_API_KEY, OPENAI_API_KEY
import pandas as pd

# ---------- SETUP ----------

# Use OpenAI for both embeddings and the LLM
llm = ChatOpenAI(model="gpt-4o",openai_api_key=OPENAI_API_KEY)
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ---------- API & HELPERS ----------
def get_jsonparsed_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return []

def get_earnings_call(symbol, year, quarter):
    url = f"https://discountingcashflows.com/api/transcript/?ticker={symbol}&quarter={quarter}&year={year}&key={DCF_API_KEY}"
    data = get_jsonparsed_data(url)
    return data[0]['content'] if data and isinstance(data, list) else None

def chunk_transcript(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def create_vectorstore(chunks):
    return FAISS.from_texts(chunks, embedding_model)

def generate_summary(chunks):
    chunk_summaries = []

    # Summarize each chunk individually
    for chunk in chunks:
        prompt = f"""
        Summarize the following part of an earnings call transcript. Include key points, risks, challenges, and opportunities.

        Transcript chunk:
        {chunk}
        """
        # Summarize using OpenAI
        chunk_summary = llm.predict(prompt)
        chunk_summaries.append(chunk_summary)

    # Combine the chunk summaries
    combined_summaries = "\n\n".join(chunk_summaries)

    # Now, create the final overall summary
    final_prompt = f"""
    Based on the following summarized content from the earnings call, provide a high-level analysis:

    1. Overall sentiment (Positive, Negative, Neutral)
    2. Five key takeaways
    3. Risks mentioned
    4. Challenges faced by the company
    5. Opportunities ahead

    Summaries:
    {combined_summaries}
    """
    # Final summary
    final_summary = llm.predict(final_prompt)
    return final_summary

def ask_question(query, vectorstore):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff"
    )
    return qa.run(query)

# ---------- STREAMLIT UI ----------
st.title('Earnings Call Transcript Analyzer')

ticker = st.text_input("Enter company ticker symbol:", value="AAPL").upper()
col1, col2 = st.columns(2)
with col1:
    year = st.selectbox("Select year:", options=list(range(2025, 2015, -1)))
with col2:
    quarter = st.selectbox("Select quarter:", options=["Q1", "Q2", "Q3", "Q4"])

if st.button('Fetch Transcript'):
    with st.spinner("Fetching transcript..."):
        transcript = get_earnings_call(ticker, year, quarter)
        if transcript:
            st.success("Transcript fetched successfully.")
            st.session_state.transcript = transcript
        else:
            st.warning("Transcript not found.")

if 'transcript' in st.session_state:
    st.subheader("Transcript Preview")
    # Create a scrollable container using markdown and custom CSS
    # Replace \n with <br> to create extra line breaks between paragraphs
    formatted_transcript = st.session_state.transcript.replace("\n", "<br><br>")  # Adding space between paragraphs

    st.markdown(f"""
    <div style="max-height: 400px; overflow-y: scroll;">
        {formatted_transcript}
    </div>
    """, unsafe_allow_html=True)

    chunks = chunk_transcript(st.session_state.transcript)
    vectorstore = create_vectorstore(chunks)

    # if st.button("🧠 Generate Summary"):
    #     with st.spinner("Analyzing..."):
    #         summary = generate_summary(chunks)
    #         st.subheader("📌 Summary")
    #         st.markdown

    st.subheader("Ask a Question")
    user_q = st.text_input("Your question:")
    if st.button("Ask"):
        if user_q:
            with st.spinner("Thinking..."):
                answer = ask_question(user_q, vectorstore)
                st.markdown(f"**Answer:** {answer}")
        else:
            st.warning("Please enter a question.")