import json
from collections import Counter
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

def load_documents(sample_size=6000):
      dataset = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
        data_home=str(ARTIFACTS / "sklearn_cache"),
    )
      documents = []
      for source_id, (text,label) in enumerate(
          zip(dataset.data, dataset.target)
      ):
          text = text.strip()
          if not text:
              continue
          documents.append({
            "id": source_id,
            "text": text,
            "category": dataset.target_names[int(label)],
        })
          if len(documents) == sample_size:
              break
      if len(documents) < sample_size:
          raise ValueError("Not enough nonempty documents.")
      return documents



def main():
   ARTIFACTS.mkdir(parents=True, exist_ok=True)
   documents = load_documents()

   output_path = ARTIFACTS / "documents.json"

   output_path.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
   print(f"Saved {len(documents)} documents to {output_path}")
   print("Category counts:")

   for category, count in sorted(
       Counter(doc["category"] for doc in documents).items()
   ):
       print(f"  {category}: {count}")
   
   print("\nFirst document preview:")
   print(documents[0]["text"][:300])

if __name__ == "__main__":
   main()
