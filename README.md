# Multi-Agent Explainable LLM Fact Verification System

An explainable AI-powered fact verification system that combines **Retrieval-Augmented Generation (RAG), semantic search, Natural Language Inference (NLI), and multi-agent reasoning** to evaluate factual claims using supporting evidence.

The system retrieves relevant information from a knowledge base, analyzes evidence from multiple perspectives, and generates an evidence-backed verdict with an explanation. Its goal is to reduce unsupported LLM-generated responses and improve the transparency of automated fact verification.

## Key Features

- **Evidence Retrieval:** Uses BGE embeddings and FAISS for semantic similarity-based evidence retrieval.
- **Cross-Encoder Reranking:** Improves the relevance ordering of retrieved evidence.
- **Multi-Agent Reasoning:** Uses PRO, CON, and NEUTRAL agents to analyze claims from different perspectives.
- **LLM-Based Judgement:** A JUDGE agent evaluates the findings and produces a final assessment.
- **Natural Language Inference:** Identifies whether evidence supports, contradicts, or is neutral toward a claim.
- **Explainable Results:** Provides a verdict with an evidence-based explanation.
- **Performance Evaluation:** Evaluates retrieval performance and generated explanations using established metrics.

## System Architecture

```text
              User Input (Fact / Claim)
                       |
                       v
               BGE Embeddings
                       |
                       v
                FAISS Retrieval
                       |
                       v
             Cross-Encoder Reranking
                       |
                       v
             Relevant Evidence
                       |
          +------------+------------+
          |            |            |
          v            v            v
       PRO Agent    CON Agent   NEUTRAL Agent
          |            |            |
          +------------+------------+
                       |
                       v
                  JUDGE Agent
                       |
                       v
             Natural Language Inference
                       |
                       v
              Final Verdict & Explanation
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| LLMs | Gemini, Llama 3, Claude |
| Embeddings | BGE |
| Vector Search | FAISS |
| Reranking | Cross-Encoder |
| Architecture | Retrieval-Augmented Generation (RAG) |
| Reasoning | Multi-Agent System |
| Verification | Natural Language Inference (NLI) |
| Dataset / Knowledge Source | FEVER, Wikipedia |
| Interface | Streamlit |

## How It Works

### 1. Claim Input
The user submits a factual statement that needs to be verified.

### 2. Semantic Embedding
The BGE embedding model converts the input claim into a numerical vector representation that captures its semantic meaning.

### 3. Evidence Retrieval
FAISS performs similarity-based vector search against the knowledge base and retrieves relevant evidence passages.

### 4. Evidence Reranking
A Cross-Encoder evaluates the relationship between the claim and retrieved passages to improve their relevance ranking.

### 5. Multi-Agent Analysis

The retrieved evidence is analyzed by three specialized agents:

- **PRO Agent:** Identifies evidence supporting the given claim.
- **CON Agent:** Identifies evidence contradicting the claim.
- **NEUTRAL Agent:** Examines ambiguous, inconclusive, or insufficient evidence.

### 6. Final Judgement
The JUDGE agent evaluates the findings from the three agents and uses Natural Language Inference to assess the relationship between the evidence and the original claim.

### 7. Verdict Generation
The system generates a final assessment along with an explanation grounded in the retrieved evidence.

## Performance Metrics

| Metric | Result |
|---|---:|
| Evidence Retrieval Hit@5 | 97.30% |
| Average Retrieval Time | 0.0161 seconds |
| BERTScore | 0.8217 |
| ROUGE-L | 0.3556 |
| METEOR | 0.2927 |

### Metric Description

- **Hit@5:** Measures whether at least one relevant evidence passage is retrieved within the top five results.
- **Retrieval Time:** Measures the reported average time for the retrieval operation.
- **BERTScore:** Evaluates semantic similarity between generated and reference explanations.
- **ROUGE-L:** Measures longest common subsequence-based overlap.
- **METEOR:** Evaluates generated text using lexical matching and linguistic alignment.

*Note: Hit@5 evaluates evidence retrieval, not final fact-verification accuracy. BERTScore, ROUGE-L, and METEOR evaluate generated text rather than directly establishing factual correctness.*

## Dataset

The project uses:

- **FEVER Dataset:** A fact-verification benchmark containing claims and evidence annotations.
- **Wikipedia:** A source of textual evidence for claim verification.

## Applications

- Automated fact checking
- Misinformation and claim analysis
- Evidence-based question answering
- LLM hallucination analysis
- Explainable AI applications
- Research and information verification

## Future Enhancements

- Support for multilingual and regional-language fact verification.
- Integration of additional trusted knowledge sources.
- Improved confidence calibration and evidence attribution.
- Advanced retrieval optimization for larger knowledge bases.
- More comprehensive evaluation of factual consistency and agent reliability.

## Conclusion

This project demonstrates how Retrieval-Augmented Generation and multi-agent reasoning can be combined to build a more transparent fact-verification workflow. By retrieving relevant evidence, analyzing claims from multiple perspectives, and generating explainable assessments, the system aims to make LLM-based information verification more reliable.

---

**Core Concepts:** Multi-Agent LLMs | RAG | NLP | Semantic Search | FAISS | BGE Embeddings | Cross-Encoder Reranking | NLI | Explainable AI
