
from sentence_transformers import SentenceTransformer, util

from extractor import extract_pdf, generate_gemini_response


# Load the embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# RAG configuration
CHUNK_SIZE = 180
CHUNK_OVERLAP = 40
TOP_K = 3


def create_chunks(pages):
    """
    Split PDF text into smaller overlapping chunks.
    Keep the original page number for source citations.
    """

    chunks = []

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        words = text.split()

        start = 0

        while start < len(words):

            end = min(start + CHUNK_SIZE, len(words))

            chunk_text = " ".join(words[start:end])

            if chunk_text.strip():

                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

            if end == len(words):
                break

            start = end - CHUNK_OVERLAP

    return chunks


def answer_question(pdf_bytes, question):

    # 1. Extract PDF text
    pages = extract_pdf(pdf_bytes)

    # 2. Split pages into overlapping chunks
    chunks = create_chunks(pages)

    if not chunks:
        return (
            "I could not find readable text in this PDF. "
            "It may be a scanned document."
        )

    # 3. Embed all chunks
    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # 4. Embed the user's question
    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # 5. Rank chunks by semantic similarity
    similarity_scores = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    top_k = min(TOP_K, len(chunks))

    top_results = similarity_scores.topk(top_k)

    retrieved_chunks = []

    for score, index in zip(
        top_results.values.tolist(),
        top_results.indices.tolist()
    ):

        chunk = chunks[index]

        retrieved_chunks.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "score": score
        })

    # 6. Build context with page references
    context_parts = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
            Source {number}
            PDF page: {chunk["page"]}
            Similarity score: {chunk["score"]:.3f}
            Text:
            {chunk["text"]}
            """
        )

    context = "\n".join(context_parts)

    # 7. Ask Gemini to answer using retrieved context
    prompt = f"""
    You are OpportunityLens, an internship information assistant.

    Answer the user's question using only the supplied context.

    Rules:
    1. Do not invent facts or use outside knowledge.
    2. If the context does not contain the answer, say:
       "I could not find this information in the uploaded PDF."
    3. Cite supporting PDF page numbers in your answer,
       for example: [PDF page 2].
    4. If multiple sources support the answer, cite each
       relevant page.
    5. Similarity scores are retrieval scores, not proof
       that a statement is correct.

    Retrieved context:
    {context}

    User question:
    {question}
    """

    return generate_gemini_response(prompt)
