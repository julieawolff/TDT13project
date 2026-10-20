"""
Steg 6: tekstmodell med Sentence-BERT-embeddinger i stedet for TF-IDF.

Samme idé som text_baseline.py (slå sammen tekst per bruker, tren en
regresjon per Big Five-trekk, evaluer med 5-fold kryssvalidering),
men her bruker vi en FERDIGTRENT transformer-modell til å gjøre om
tekst til tall, i stedet for å telle ord selv (TF-IDF).

VIKTIG FORSKJELL fra TF-IDF-skriptet:
TF-IDF måtte "læres" (fit) på treningsdataene i hver fold, fordi
ordforrådet kommer FRA dataene våre. Sentence-BERT er derimot en
ferdigtrent modell -- den er laget av andre, på helt andre tekster,
og lærer ingenting av VÅRE data når den lager embeddinger. Det betyr
at vi trygt kan regne ut embeddingene for ALLE brukere én gang, før
vi i det hele tatt begynner med kryssvalideringen. Det er ingen fare
for data leakage her, og det sparer oss for mye regnetid (å kjøre
transformeren er det desidert tyngste steget).

Installer først (i venv-et ditt):
    pip install sentence-transformers

Kjør:
    python text_sbert.py
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import os

BIG5_COLS = ["agreeableness", "openness", "conscientiousness",
             "extraversion", "neuroticism"]

N_FOLDS = 5
RANDOM_SEED = 42

# Maks antall kommentarer vi bruker PER BRUKER når vi lager
# embeddingen. Noen brukere har over 50 000 kommentarer (se
# utforskingen fra tidligere) -- å kjøre transformeren på alle ville
# tatt svært lang tid, og gir trolig ikke noe bedre resultat enn et
# representativt utvalg. Vi trekker et tilfeldig utvalg per bruker.
MAX_COMMENTS_PER_USER = 200

# Navnet på selve embedding-modellen. "all-MiniLM-L6-v2" er en liten,
# rask og mye brukt Sentence-BERT-modell -- gir god kvalitet uten å
# kreve GPU.
MODEL_NAME = "all-MiniLM-L6-v2"

# Vi lagrer embeddingene til disk etter første kjøring, slik at du
# slipper å vente på at transformeren kjører på nytt hver gang du
# justerer noe lenger ned i skriptet (f.eks. regresjonsmodellen).
EMBEDDINGS_CACHE = "user_embeddings.npz"


def build_user_embeddings(comments_path="comments_clean.csv",
                           profiles_path="profiles_clean.csv"):
    """
    Lager én embedding-vektor per bruker: gjennomsnittet av
    embeddingene til et utvalg av kommentarene deres.
    """
    if os.path.exists(EMBEDDINGS_CACHE):
        print(f"Fant mellomlagrede embeddinger i {EMBEDDINGS_CACHE}, "
              f"laster disse i stedet for å regne på nytt.")
        cached = np.load(EMBEDDINGS_CACHE, allow_pickle=True)
        return cached["authors"], cached["embeddings"], pd.read_csv(profiles_path)

    comments = pd.read_csv(comments_path)
    profiles = pd.read_csv(profiles_path)

    print(f"Laster embedding-modellen ({MODEL_NAME})... "
          f"(lastes ned automatisk første gang)")
    model = SentenceTransformer(MODEL_NAME)

    authors = []
    embeddings = []

    # Går gjennom én bruker om gangen
    grouped = comments.groupby("author")
    n_users = comments["author"].nunique()

    for i, (author, group) in enumerate(grouped):
        # Trekk et tilfeldig utvalg av kommentarene til denne
        # brukeren, maks MAX_COMMENTS_PER_USER stykker. random_state
        # gjør at utvalget blir det samme hver gang du kjører.
        sample = group["body"].sample(
            n=min(len(group), MAX_COMMENTS_PER_USER),
            random_state=RANDOM_SEED,
        ).astype(str).tolist()

        # encode() gjør om en liste med tekster til en matrise med
        # tall (én rad med ca. 384 tall per kommentar for denne
        # modellen). show_progress_bar=False fordi vi already printer
        # fremdrift selv, under.
        comment_embeddings = model.encode(sample, show_progress_bar=False,
                                           batch_size=64)

        # Gjennomsnittet av alle kommentar-embeddingene til brukeren
        # blir brukerens samlede "tekst-signatur".
        user_embedding = comment_embeddings.mean(axis=0)

        authors.append(author)
        embeddings.append(user_embedding)

        if (i + 1) % 100 == 0 or (i + 1) == n_users:
            print(f"  ...behandlet {i + 1}/{n_users} brukere")

    authors = np.array(authors)
    embeddings = np.vstack(embeddings)

    # Lagre til disk, så vi slipper å regne dette på nytt neste gang
    np.savez(EMBEDDINGS_CACHE, authors=authors, embeddings=embeddings)
    print(f"\nLagret embeddinger til {EMBEDDINGS_CACHE}")

    return authors, embeddings, profiles


def evaluate(y_true, y_pred, trait_names):
    results = []
    for i, trait in enumerate(trait_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
        results.append({"trait": trait, "MAE": mae, "Pearson_r": corr})
    return pd.DataFrame(results)


def run_cross_validation(X, y):
    """
    Samme kryssvalideringsoppsett som i text_baseline.py: 5 folder,
    hver bruker er testbruker nøyaktig én gang.
    """
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold_num, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        print(f"\n=== Fold {fold_num}/{N_FOLDS} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Merk: her er det INGEN vectorizer å trene -- embeddingene
        # er allerede ferdige tall, regnet ut av den ferdigtrente
        # Sentence-BERT-modellen. Vi trenger bare regresjonen.
        predictions = np.zeros_like(y_test, dtype=float)
        for i, trait in enumerate(BIG5_COLS):
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train[:, i])
            predictions[:, i] = model.predict(X_test)

        fold_result = evaluate(y_test, predictions, BIG5_COLS)
        fold_result["fold"] = fold_num
        fold_results.append(fold_result)

    return pd.concat(fold_results)


def summarize(all_folds, label):
    summary = (
        all_folds.groupby("trait")[["MAE", "Pearson_r"]]
        .agg(["mean", "std"])
    )
    print(f"\n--- {label} (gjennomsnitt ± std over {N_FOLDS} folder) ---")
    print(summary.to_string())
    return summary


def main():
    authors, embeddings, profiles = build_user_embeddings()

    # Koble embeddingene (etter author) sammen med Big Five-skårene,
    # i riktig rekkefølge
    embed_df = pd.DataFrame({"author": authors})
    embed_df = embed_df.merge(
        profiles[["author"] + BIG5_COLS], on="author", how="inner"
    )

    # Filtrer embeddings-matrisen til kun de brukerne som faktisk
    # matchet (bør være alle, men trygt å sjekke)
    keep_mask = np.isin(authors, embed_df["author"].values)
    X = embeddings[keep_mask]
    y = embed_df[BIG5_COLS].values

    print(f"\n{len(X)} brukere klare for modellering "
          f"(embedding-dimensjon: {X.shape[1]})")

    model_folds = run_cross_validation(X, y)
    summarize(model_folds, "Sentence-BERT + Ridge-regresjon (tekstmodellen)")


if __name__ == "__main__":
    main()