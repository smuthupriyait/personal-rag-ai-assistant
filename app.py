import streamlit as st
from sentence_transformers import SentenceTransformer

from src.config import DISTANCE_THRESHOLD, EMBEDDING_MODEL
from src.generation.generator import generate_answer, prepare_context
from src.retrieval.search import load_vector_store, search


st.set_page_config(
    page_title="NordicTech RAG Assistant",
    page_icon="🤖",
    layout="centered"
)


@st.cache_resource
def load_rag_components():
    model = SentenceTransformer(EMBEDDING_MODEL)
    index, chunks = load_vector_store()

    return model, index, chunks


st.title("🤖 NordicTech RAG Assistant")

st.write(
    "Ask questions about the NordicTech company documents."
)

st.divider()


with st.form("question_form"):

    question = st.text_input(
        "Your question",
        placeholder="e.g. How many vacation days do employees get?"
    )

    ask_button = st.form_submit_button(
        "Ask",
        type="primary"
    )


if ask_button:

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            model, index, chunks = load_rag_components()

            retrieved_chunks = search(
                question,
                model,
                index,
                chunks,
                distance_threshold=DISTANCE_THRESHOLD
            )

            if not retrieved_chunks:

                st.warning(
                    "I don't have enough information in the "
                    "NordicTech documents to answer that question."
                )

            else:

                context = prepare_context(retrieved_chunks)

                answer = generate_answer(
                    question,
                    context
                )

                st.subheader("Answer")

                st.write(answer)

                st.subheader("Sources")

                for chunk in retrieved_chunks:

                    source_name = (
                        f"{chunk['filename']} — "
                        f"{chunk['section']}"
                    )

                    with st.expander(source_name):

                        st.write(chunk["content"])