

import streamlit as st
import json
import pickle
import faiss
import numpy as np
import re
import spacy

from openai import OpenAI

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity




st.set_page_config(

    page_title="Explainable Fact Verification",

    page_icon="",

    layout="wide"
)



st.markdown("""

<style>

html, body, [class*="css"]  {
    background: linear-gradient(135deg, #020617, #0f172a, #111827);
    color: white;
}

.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #111827);
    color: white;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.main-title {
    text-align: center;
    font-size: 46px;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 35px;
}

.evidence-card {
    background: rgba(255,255,255,0.05);
    border-left: 4px solid #38bdf8;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 14px;
}

.agent-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 18px;
    margin-top: 10px;
}

.agent-pro {
    border-left: 5px solid #22c55e;
}

.agent-con {
    border-left: 5px solid #ef4444;
}

.agent-neutral {
    border-left: 5px solid #facc15;
}

.judge-card {
    background: rgba(59,130,246,0.15);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(99,102,241,0.3);
}

</style>

""", unsafe_allow_html=True)




st.markdown(

    """
    <div class="main-title">
    Multi-Agent Explainable Fact Verification
    </div>
    """,

    unsafe_allow_html=True
)

st.markdown(

    """
    <div class="subtitle">
    Hybrid Retrieval • Multi-Agent Reasoning • Multi-LLM Comparison
    </div>
    """,

    unsafe_allow_html=True
)




with st.sidebar:

    st.header("LLM Applications")

    st.markdown("""

    • Multi-Agent Reasoning  
    • Explainable AI  
    • Fact Verification  
    • Evidence-Based Reasoning  
    • Hallucination Detection  
    • Multi-LLM Comparison  
    • NLP + RAG Pipeline  

    """)


    selected_models = st.multiselect(

        "Select LLM Models",

        [

            "Gemini Flash",

            "Llama 3",

            "Claude"
        ],

        default=[

            "Gemini Flash",

            "Llama 3",

            "Claude"
        ]
    )



OPENROUTER_API_KEY = "###########################"

client = OpenAI(

    api_key=OPENROUTER_API_KEY,

    base_url="https://openrouter.ai/api/v1"
)



llm_models = {

    "Gemini Flash":
    "google/gemini-2.0-flash-001",

    "Llama 3":
    "meta-llama/llama-3-8b-instruct",

    "Claude":
    "anthropic/claude-3-haiku"
}



@st.cache_resource
def load_spacy():

    return spacy.load(
        "en_core_web_sm"
    )

nlp = load_spacy()


@st.cache_resource
def load_sbert():

    return SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

retriever_model = load_sbert()


@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

reranker = load_reranker()


@st.cache_resource
def load_verifier():

    return pipeline(

        "text-classification",

        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",

        truncation=True
    )

verifier = load_verifier()



@st.cache_resource
def load_files():

    with open(

        r"C:\Users\rishi\PycharmProjects\nllp\wiki_metadata.pkl",

        "rb"

    ) as f:

        wiki_metadata = pickle.load(f)

    embeddings = np.load(

        r"C:\Users\rishi\PycharmProjects\nllp\hybrid_embeddings.npy"
    )

    index = faiss.read_index(

        r"C:\Users\rishi\PycharmProjects\nllp\hybrid_faiss.index"
    )

    return (

        wiki_metadata,

        embeddings,

        index
    )


wiki_metadata, embeddings, index = load_files()



def generate_response(

    prompt,

    model_name
):

    try:

        completion = client.chat.completions.create(

            model=model_name,

            messages=[

                {

                    "role": "user",

                    "content": prompt
                }
            ],

            temperature=0.3,

            max_tokens=512
        )

        return completion.choices[0].message.content

    except Exception as e:

        return f"ERROR: {str(e)}"



def retrieve_evidence(

    claim,

    top_k=5
):

    query_embedding = retriever_model.encode(
        [claim]
    )

    distances, indices = index.search(

        np.array(query_embedding).astype("float32"),

        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(
            wiki_metadata[idx]
        )

    return results


def rerank_evidence(

    claim,

    evidence_list
):

    pairs = []

    for item in evidence_list:

        pairs.append(
            (claim, item["text"])
        )

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(

        zip(scores, evidence_list),

        reverse=True,

        key=lambda x: x[0]
    )

    final_evidence = []

    for score, item in ranked:

        item["rerank_score"] = float(score)

        final_evidence.append(item)

    return final_evidence




def verify_claim_nli(

    claim,

    top_evidence
):

    label_map = {

        "entailment": "SUPPORTS",

        "contradiction": "REFUTES",

        "neutral": "NOT ENOUGH INFO"
    }

    final_scores = {

        "SUPPORTS": 0,

        "REFUTES": 0,

        "NOT ENOUGH INFO": 0
    }

    for item in top_evidence[:2]:

        evidence = item["text"]

        final_input = evidence + " [SEP] " + claim

        result = verifier(
            final_input
        )

        pred = result[0]

        label = pred["label"].lower()

        score = pred["score"]

        mapped_label = label_map[label]

        final_scores[mapped_label] = max(

            final_scores[mapped_label],

            score
        )

    predicted_label = max(

        final_scores,

        key=final_scores.get
    )

    confidence = round(

        max(final_scores.values()) * 100,

        2
    )

    return predicted_label, confidence



def pro_agent(

    claim,

    evidence,

    model_name
):

    prompt = f"""
Support the claim using evidence.

Claim:
{claim}

Evidence:
{evidence}
"""

    return generate_response(
        prompt,
        model_name
    )


def con_agent(

    claim,

    evidence,

    model_name
):

    prompt = f"""
Refute the claim using evidence.

Claim:
{claim}

Evidence:
{evidence}
"""

    return generate_response(
        prompt,
        model_name
    )


def neutral_agent(

    claim,

    evidence,

    model_name
):

    prompt = f"""
Determine whether evidence is insufficient.

Claim:
{claim}

Evidence:
{evidence}
"""

    return generate_response(
        prompt,
        model_name
    )


def judge_agent(

    claim,

    evidence,

    pro_output,

    con_output,

    neutral_output,

    model_name
):

    prompt = f"""
You are the final judge.

Claim:
{claim}

Evidence:
{evidence}

PRO:
{pro_output}

CON:
{con_output}

NEUTRAL:
{neutral_output}

Return:
- Final Reasoning
"""

    return generate_response(
        prompt,
        model_name
    )


claim = st.text_input(
    "Enter Claim"
)



if st.button("Verify Claim"):

    if claim.strip() == "":

        st.warning("Please enter a claim.")

    else:

        with st.spinner(
            "Running Multi-Agent Verification..."
        ):

            retrieved = retrieve_evidence(
                claim
            )

            top_evidence = rerank_evidence(

                claim,

                retrieved
            )

            prediction, confidence = verify_claim_nli(

                claim,

                top_evidence
            )

            evidence_text = ""

            for i, item in enumerate(top_evidence):

                evidence_text += (

                    f"Evidence {i+1}: "

                    + item["text"]

                    + "\n\n"
                )



        st.header("Retrieved Evidence")

        for i, item in enumerate(top_evidence):

            st.markdown(

                f"""
                <div class="evidence-card">
                <b>Evidence {i+1}</b><br><br>
                {item["text"]}
                </div>
                """,

                unsafe_allow_html=True
            )


        for model_title in selected_models:

            model_name = llm_models[
                model_title
            ]

            st.divider()

            st.header(f"{model_title}")

            with st.spinner(

                f"Running {model_title}..."
            ):

                pro_output = pro_agent(

                    claim,

                    evidence_text,

                    model_name
                )

                con_output = con_agent(

                    claim,

                    evidence_text,

                    model_name
                )

                neutral_output = neutral_agent(

                    claim,

                    evidence_text,

                    model_name
                )

                judge_output = judge_agent(

                    claim,

                    evidence_text,

                    pro_output,

                    con_output,

                    neutral_output,

                    model_name
                )


            pro_output = pro_output.replace(
                "<div>", ""
            ).replace(
                "</div>", ""
            )

            con_output = con_output.replace(
                "<div>", ""
            ).replace(
                "</div>", ""
            )

            neutral_output = neutral_output.replace(
                "<div>", ""
            ).replace(
                "</div>", ""
            )

            judge_output = judge_output.replace(
                "<div>", ""
            ).replace(
                "</div>", ""
            )



            tab1, tab2, tab3, tab4 = st.tabs([

                "PRO",

                "CON",

                "NEUTRAL",

                "JUDGE"
            ])

            with tab1:

                st.markdown(

                    f'''
                    <div class="agent-card agent-pro">
                    {pro_output}
                    </div>
                    ''',

                    unsafe_allow_html=True
                )

            with tab2:

                st.markdown(

                    f'''
                    <div class="agent-card agent-con">
                    {con_output}
                    </div>
                    ''',

                    unsafe_allow_html=True
                )

            with tab3:

                st.markdown(

                    f'''
                    <div class="agent-card agent-neutral">
                    {neutral_output}
                    </div>
                    ''',

                    unsafe_allow_html=True
                )

            with tab4:

                st.markdown(

                    f'''
                    <div class="judge-card">
                    {judge_output}
                    </div>
                    ''',

                    unsafe_allow_html=True
                )