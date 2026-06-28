import streamlit as st
import tempfile
import os
from rag_engine import load_pdf, chunk_text, create_vectorstore, build_qa_chain

st.set_page_config(page_title="RAG Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Reading and indexing your PDF..."):
        pages = load_pdf(tmp_path)
        chunks = chunk_text(pages)
        vectorstore = create_vectorstore(chunks)
        chain, retriever = build_qa_chain(vectorstore)

    st.success(f"✅ Done! {len(chunks)} chunks indexed from {len(pages)} pages.")

    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
            source_docs = retriever.invoke(question)

        st.write("### 💬 Answer")
        st.write(answer)

        st.write("### 📌 Sources Used")
        for i, doc in enumerate(source_docs):
            with st.expander(f"Source {i+1} — Page {doc.metadata.get('page', '?') + 1}"):
                st.write(doc.page_content)