
import sys
import pymupdf
from sentence_transformers import util

from rag import create_chunks, embedding_model


def evaluate_retrieval(pdf_path, test_cases, k=3):

    # Extract PDF pages
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf):
            pages.append({
                "page": page_number + 1,
                "text": page.get_text()
            })

    # Use the same chunking as our RAG pipeline
    chunks = create_chunks(pages)

    if not chunks:
        print("No readable text found in this PDF.")
        return

    # Embed all chunks once
    texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = embedding_model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    recall_scores = []
    reciprocal_ranks = []

    print("\nRAG RETRIEVAL EVALUATION")
    print("=" * 45)

    for case in test_cases:

        question = case["question"]
        relevant_pages = set(case["relevant_pages"])

        query_embedding = embedding_model.encode(
            question,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        scores = util.cos_sim(
            query_embedding,
            chunk_embeddings
        )[0]

        top_indices = scores.topk(
            min(k, len(chunks))
        ).indices.tolist()

        retrieved_chunks = [
            chunks[index] for index in top_indices
        ]

        # A chunk is relevant when its page is annotated
        # as containing the answer.
        relevant_ranks = [
            rank
            for rank, chunk in enumerate(
                retrieved_chunks, start=1
            )
            if chunk["page"] in relevant_pages
        ]

        recall = 1.0 if relevant_ranks else 0.0
        reciprocal_rank = (
            1.0 / relevant_ranks[0]
            if relevant_ranks else 0.0
        )

        recall_scores.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuestion: {question}")
        print(f"Expected answer page(s): {sorted(relevant_pages)}")
        print(
            "Retrieved pages:",
            [chunk["page"] for chunk in retrieved_chunks]
        )
        print(f"Recall@{k}: {recall:.3f}")
        print(f"Reciprocal Rank: {reciprocal_rank:.3f}")

    mean_recall = sum(recall_scores) / len(recall_scores)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n" + "=" * 45)
    print(f"Mean Recall@{k}: {mean_recall:.3f}")
    print(f"MRR: {mrr:.3f}")
    print(f"Questions evaluated: {len(test_cases)}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            'Usage: python evaluate_rag.py "path\\to\\internship.pdf"'
        )
        sys.exit(1)

    # Replace these examples with questions and correct
    # page numbers from YOUR internship PDF.
    test_cases = [
        {
            "question": "What is the internship stipend?",
            "relevant_pages": [1]
        },
        {
            "question": "What are the eligibility requirements?",
            "relevant_pages": [1, 2]
        },
        {
            "question": "What is the application deadline?",
            "relevant_pages": [2]
        },
        {
            "question": "What is the duration of the internship?",
            "relevant_pages": [1]
        },
        {
            "question": "What skills or qualifications are required?",
            "relevant_pages": [1, 2]
        }
    ]

    evaluate_retrieval(
        sys.argv[1],
        test_cases,
        k=3
    )
