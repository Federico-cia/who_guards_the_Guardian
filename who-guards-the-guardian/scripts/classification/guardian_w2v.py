import ast
import pandas as pd
from gensim.models import Word2Vec

INPUT_FILE = "articles_clean_with_sentence_lemma.csv"
MODEL_FILE = "guardian_word2vec.model"

# Load input file
print("Loading data...", flush=True)
df = pd.read_csv(INPUT_FILE)

# Convert token column from string to Python lists, since .csv files gives lists as string
print("Parsing token column...", flush=True)
df["body_sent_lemma_tokens"] = df["body_sent_lemma_tokens"].apply(ast.literal_eval)

# Build the Word2Vec training corpus: one list of tokens per sentence
print("Building sentence corpus...", flush=True)
sentences = [
    sent
    for article in df["body_sent_lemma_tokens"]
    for sent in article
    if isinstance(sent, list) and len(sent) > 0
]

# Check that corpus is not empty
print(f"Number of sentences: {len(sentences)}", flush=True)
if len(sentences) == 0:
    raise ValueError("No sentences found for Word2Vec training.")

# Train a Word2Vec model on the Guardian sentence-level lemma corpus
print("Training Word2Vec...", flush=True)
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=5,
)

# Save the trained Word2Vec model
print("Saving Word2Vec model...", flush=True)
w2v_model.save(MODEL_FILE)

print("Done.", flush=True)
print(f"Model saved to: {MODEL_FILE}", flush=True)
print(
    "Gensim may also save companion NumPy files next to the model file "
    "(for example .wv.vectors.npy and .syn1neg.npy).",
    flush=True,
)