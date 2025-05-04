from pathlib import Path
from services.data_manager import load_data, local_caching
from services.sbert_engine import sbert_embedder, sbert_retriever
import pprint


def main():
    file_path = Path("sample_files/sample_archer_policies.csv")
    sheet_name = None  # Change if using XLSX
    selected_columns = ["Policy Name", "Description"]

    # Step 1: Generate file ID
    file_id = local_caching.generate_file_id(file_path)

    # Step 2: Extract data & construct sentences
    records = load_data.extract_data(file_path, sheet_name, selected_columns)
    sentences = [
        " ".join([str(r[col]) for col in selected_columns if r.get(col)])
        for r in records
    ]

    # Step 3: Embed sentences
    model_key = "MiniLM-L6-v2"
    embeddings = sbert_embedder.embed_sentences(sentences, model_key)

    # Step 4: Cache everything
    local_caching.save_records(file_id, records)
    local_caching.save_sentences(file_id, sentences)
    local_caching.save_embeddings(file_id, embeddings)
    local_caching.save_metadata(
        file_id,
        local_caching.FileMetadata(
            file_id=file_id,
            file_name=file_path.name,
            model_key=model_key,
            columns=selected_columns,
            sheet_name=sheet_name,
        ),
    )

    print(f"✅ Corpus embedded and cached under file_id: {file_id}")

    # Step 5: Accept a simple query
    query = input("\n🔍 Enter a query: ")
    query_emb = sbert_embedder.embed_query(query, model_key)

    # Step 6: Load corpus embeddings
    corpus_embs = local_caching.load_embeddings(file_id)

    # Step 7: Semantic search
    results = sbert_retriever.get_top_cosine_matches(query_emb, corpus_embs, top_k=3)

    # Step 8: Display results
    print("\n🎯 Top matches:")
    for idx, score in results:
        print(f"\nScore: {score:.4f}")
        pprint.pprint(records[idx])


if __name__ == "__main__":
    main()
