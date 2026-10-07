import pandas as pd
import spacy
import ast

INPUT_FILE = "articles_clean.csv"
OUTPUT_FILE = "articles_clean_with_both_lemma_columns.parquet"
BATCH_SIZE = 256
N_PROCESSES = 4

# We use "en_core_web_sm", small spaCy English pipeline
print("Loading spaCy model...", flush=True)
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
nlp.max_length = 5_000_000

# We define the function that performs lemmatization
def sentence_lemmatize(texts, batch_size=256, n_process=4):
    per_article_sentences = []

    texts = texts.tolist()

    for i, doc in enumerate(
        nlp.pipe(texts, batch_size=batch_size, n_process=n_process),
        start=1
    ):
        article_sentences = []

        for sent in doc.sents:
            sent_tokens = [
                token.lemma_.lower()
                for token in sent
                if not token.is_punct
                and not token.is_space
                and token.lemma_.strip() != ""
            ]

            if sent_tokens:
                article_sentences.append(sent_tokens)

        per_article_sentences.append(article_sentences)

        if i % 5000 == 0:
            print(f"Processed {i} articles...", flush=True)

    return per_article_sentences

# Import dataset
print("Loading input dataset...", flush=True)
df = pd.read_csv(INPUT_FILE)

# Apply sentence_lemmatize function to bodyContent column
print("Starting sentence-level lemmatization...", flush=True)
df["body_sent_lemma_tokens"] = sentence_lemmatize(df["bodyContent"], batch_size=BATCH_SIZE, n_process=N_PROCESSES)

# Now we create the column made by one single flattened string of lemmas
print("Building flattened lemma column...", flush=True)
df["bodyContent_lemma"] = df["body_sent_lemma_tokens"].apply(
    lambda article: " ".join(token for sent in article for token in sent)
)

print("Saving output...", flush=True)
df.to_parquet(OUTPUT_FILE, index=False)

print(f"Done. Saved to {OUTPUT_FILE}", flush=True)