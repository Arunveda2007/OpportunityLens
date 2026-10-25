 OpportunityLens 🔎
 AI-Powered Internship Intelligence Using Transformers, Generative AI & RAG

Core technologies: Transformer Models · NLP · Vector Embeddings · Semantic Search · Retrieval-Augmented Generation (RAG) · Large Language Models (LLMs) · Generative AI**

Choosing an internship involves more than just finding an interesting job description. Important details, eligibility requirements, and application information can be buried inside lengthy documents.

I built **OpportunityLens** to explore how modern AI techniques can make this process easier. The application extracts information from internship PDFs, answers questions using document context, and analyses eligibility against a student's profile.

What interested me most about this project was bringing different components together — transformer-based embeddings, semantic retrieval, and LLM generation — to build a working AI application rather than using a language model in isolation.

 What can it do?

*Internship Information Extraction:** Uses Google Gemini to extract structured information from internship documents.
*Transformer-Based Semantic Search:** Uses Sentence Transformers to represent text as embeddings and find relevant passages based on meaning.
*RAG-Based Question Answering:** Retrieves relevant PDF passages before sending context to an LLM to generate an answer.
*Source-Grounded Responses:** Instructs the LLM to answer from retrieved context and provide PDF page references.
*Eligibility Analysis:** Uses Gemini to compare internship requirements with a student's profile.

How the RAG pipeline works

The question-answering workflow combines information retrieval with generative AI.

```text
             Internship PDF
                   |
                   v
          PDF Text Extraction
                PyMuPDF
                   |
                   v
             Text Chunking
       180 words / 40-word overlap
                   |
                   v
        Transformer-Based Embeddings
          all-MiniLM-L6-v2
                   |
                   v
           Semantic Retrieval
          Cosine Similarity
           Top-3 Chunks
                   |
                   v
       Retrieved Context + Question
                   |
                   v
              Gemini LLM
                   |
                   v
       Context-Grounded Answer
        with PDF Page References
```

# 1. Document processing

The application extracts text from uploaded PDFs using PyMuPDF. The extracted text is divided into overlapping chunks so that individual passages can be retrieved when a user asks a question.

# 2. Transformer embeddings

OpportunityLens uses **Sentence Transformers with `all-MiniLM-L6-v2`** to convert document chunks and questions into dense vector representations.

These embeddings help the system identify passages that are semantically related to a question, even when the wording is different.

# 3. Semantic retrieval

The application compares the question embedding with document-chunk embeddings using **cosine similarity**. It selects up to three top-ranked chunks to provide context for the next stage.

# 4. Retrieval-Augmented Generation

The retrieved passages and user's question are passed to **Google Gemini**, a large language model.

The prompt instructs the model to use the supplied context, avoid inventing unsupported details, and cite relevant PDF pages. This approach is intended to make responses more traceable to the original document.

# 5. Eligibility analysis

A separate workflow uses Gemini to compare internship information with a student profile and provide an eligibility assessment.

 Technology stack

| Area | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| PDF text extraction | PyMuPDF |
| Transformer embeddings | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Semantic retrieval | Cosine similarity |
| LLM / Generative AI | Google Gemini API |
| Environment configuration | python-dotenv |
| Testing | pytest |

 Project structure

```text
OpportunityLens/
├── app.py               # Streamlit user interface
├── extractor.py         # PDF extraction and Gemini integration
├── rag.py               # Chunking, retrieval, and RAG answers
├── eligibility.py       # Internship eligibility analysis
├── evaluate_rag.py      # Retrieval evaluation
├── test_extractor.py    # Extraction tests
├── test_eligibility.py  # Eligibility tests
├── requirements.txt     # Project dependencies
├── .gitignore
└── README.md
```

Run the project locally

 Prerequisites

- Python installed on your system
- A Google Gemini API key

 1. Clone the repository

```bash
git clone https://github.com/Arunveda2007/OpportunityLens.git
cd OpportunityLens
```

 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

 3. Install dependencies

```bash
pip install -r requirements.txt
```

 4. Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key.

Security: Never commit your `.env` file or expose your API key in source code.

5. Start the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can interact with OpportunityLens.
 Retrieval evaluation

The project includes `evaluate_rag.py` to evaluate retrieval quality using two information-retrieval metrics:

- Recall@3:Measures whether relevant passages appear among the three retrieved results.
- Mean Reciprocal Rank (MRR): Measures how highly the first relevant result is ranked.

These metrics can help investigate retrieval performance and compare potential improvements. Meaningful results depend on using a properly labelled evaluation dataset and verifying the relevant-page annotations.

Benchmark scores are not reported here because they should be based on measured evaluation results.

 What I want to improve next

Building this project has made me interested in a question that goes beyond generating a fluent answer: **how can we make an AI system retrieve the right evidence and produce answers that are easier to verify?**

Some directions I want to explore include:

- Comparing embedding models and chunking strategies.
- Improving retrieval quality through reranking.
- Evaluating answer faithfulness and citation accuracy.
- Testing how retrieval errors affect the final generated answer.
- Expanding automated tests and improving error handling.

 About

I'm Arunveda, a computer science student interested in machine learning, natural language processing, and generative AI.

OpportunityLens is part of my hands-on learning journey in AI engineering, where I'm exploring how individual ML components can be combined into a useful application.

**GitHub:** [@Arunveda2007](https://github.com/Arunveda2007)
