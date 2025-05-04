from pathlib import Path
from services import semantic_search
import pprint


def main():
    file_path = Path("sample_files/sample_large_control-procedures_dataset.xlsx")
    sheet_name = "Sheet1"
    columns = ["name", "description"]
    model_key = "MiniLM-L6-v2"

    # Step 1: Prepare corpus using orchestration
    embedding_id = semantic_search.prepare_corpus(
        file_path=file_path,
        sheet_name=sheet_name,
        columns=columns,
        model_key=model_key,
    )
    print(f"✅ Corpus embedded and cached under embedding_id: {embedding_id}")

    # Step 2: Prompt user for a query
    query = input("\n🔍 Enter a query: ")

    # Step 3: Run semantic search
    results = semantic_search.query_corpus(
        query=query,
        embedding_id=embedding_id,
        model_key=model_key,
        top_k=3,
    )

    # Step 4: Display results
    print("\n🎯 Top matches:")
    for record, score in results:
        print(f"\nScore: {score:.4f}")
        pprint.pprint(record)


if __name__ == "__main__":
    main()
