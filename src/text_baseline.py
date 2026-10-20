"""
Steg 5: tekst-baseline (TF-IDF + regresjon), med 5-fold kryssvalidering.

Mål: for hver bruker, slå sammen alle kommentarene deres til én lang
tekst, og lær en modell som gjetter de fem Big Five-skårene ut fra
denne teksten.

Hvorfor kryssvalidering i stedet for én enkelt train/test-deling?
Med rundt 1300-1400 brukere blir et enkelt testsett ganske lite
(~200-270 brukere). Da kan resultatet svinge ganske mye bare basert
på hvilke brukere som tilfeldigvis havnet i testsettet den gangen.
Kryssvalidering kjører hele prosessen flere ganger, med forskjellig
test-del hver gang, og vi rapporterer gjennomsnitt og standardavvik.
Det gir et mer pålitelig og rettferdig bilde -- spesielt viktig når
du senere skal sammenligne denne tekstmodellen mot nettverksmodellen
og den kombinerte modellen.

Kjør:
    python text_baseline.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# De fem Big Five-trekkene vi skal predikere (kolonnenavn i profiles_clean.csv)
BIG5_COLS = ["agreeableness", "openness", "conscientiousness",
             "extraversion", "neuroticism"]

N_FOLDS = 5         # antall "deler" datasettet splittes i (5-fold kryssvalidering)
RANDOM_SEED = 42     # fast "frø" -- gjør at delingen blir den samme hver gang
                      # du kjører skriptet, så resultatene er reproduserbare


def load_and_aggregate(comments_path="comments_clean.csv",
                        profiles_path="profiles_clean.csv"):
    """
    Leser inn de to rensede filene, og slår sammen alle kommentarene
    til hver bruker til én lang tekst (siden vi vil ha ÉN
    tekst-representasjon per bruker, ikke per kommentar).
    """
    comments = pd.read_csv(comments_path)
    profiles = pd.read_csv(profiles_path)

    print("Slår sammen kommentarer per bruker (kan ta litt tid)...")
    # groupby("author") samler alle radene til samme bruker i én gruppe.
    # ["body"].apply(...) går gjennom hver gruppe og limer sammen all
    # teksten til én lang streng, adskilt med mellomrom.
    user_text = (
        comments.groupby("author")["body"]
        .apply(lambda texts: " ".join(texts.astype(str)))
        .reset_index()
        .rename(columns={"body": "all_text"})
    )

    # Koble sammen tekst-tabellen med Big Five-skårene, basert på
    # brukernavn ("author").
    data = user_text.merge(profiles, on="author")
    print(f"{len(data)} brukere med både tekst og Big Five-skårer")
    return data


def evaluate(y_true, y_pred, trait_names):
    """
    Regner ut to mål på hvor god modellen er, for hvert av de fem
    trekkene:

    - MAE (Mean Absolute Error): gjennomsnittlig avstand mellom
      gjettet og faktisk skåre. Lavere er bedre. Et MAE på 15 betyr
      at modellen i snitt bommer med 15 poeng (på en skala 0-100).

    - Pearson-korrelasjon: måler om gjettene "følger" de ekte
      skårene oppover og nedover, uavhengig av eksakt avstand.
      1.0 = perfekt samsvar, 0.0 = ingen sammenheng (tilfeldig gjett).
    """
    results = []
    for i, trait in enumerate(trait_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
        results.append({"trait": trait, "MAE": mae, "Pearson_r": corr})
    return pd.DataFrame(results)


def run_cross_validation(data):
    """
    Kjører hele trenings- og evalueringsprosessen N_FOLDS ganger.

    Hvordan 5-fold kryssvalidering fungerer:
    1. Del alle brukerne tilfeldig i 5 like store "folder" (grupper).
    2. Kjør 5 runder. I hver runde:
       - 4 av foldene brukes til TRENING
       - 1 fold (den som ikke ble brukt til trening) brukes til TEST
       - Hvilken fold som er "testfold" bytter for hver runde, slik
         at hver bruker til slutt har vært i testsettet nøyaktig én gang.
    3. Vi samler resultatene (MAE, Pearson) fra alle 5 rundene, og
       regner ut gjennomsnitt og standardavvik til slutt.
    """
    # KFold lager selve inndelingen i folder. shuffle=True betyr at
    # brukerne blandes tilfeldig før de deles inn (ellers ville
    # inndelingen bare fulgt rekkefølgen i filen).
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # All tekst og alle Big Five-skårer, som numpy-arrays (det
    # formatet scikit-learn forventer).
    texts = data["all_text"].values
    y = data[BIG5_COLS].values

    # Her samler vi ett resultat-datasett per fold, så vi kan slå
    # dem sammen og regne ut gjennomsnitt til slutt.
    fold_results = []

    # "baseline_results" er en enkel referanse: hva om modellen bare
    # gjetter gjennomsnittet for alle, uansett tekst? Hvis den ekte
    # modellen vår ikke er bedre enn dette, har den ikke lært noe.
    baseline_results = []

    # enumerate gir oss et fold-nummer (0, 1, 2, 3, 4) i tillegg til
    # selve indeksene. kfold.split(texts) gir oss, for hver runde,
    # hvilke rad-indekser som skal være trening og hvilke som er test.
    for fold_num, (train_idx, test_idx) in enumerate(kfold.split(texts), start=1):
        print(f"\n=== Fold {fold_num}/{N_FOLDS} ===")

        X_train_text = texts[train_idx]
        X_test_text = texts[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        print(f"  Trening: {len(X_train_text)} brukere, "
              f"Test: {len(X_test_text)} brukere")

        # TF-IDF: gjør tekst om til tall, basert på hvor ofte ord
        # forekommer. VIKTIG: vektoriseren "læres" (fit) KUN på
        # treningsdataene i denne folden. Deretter brukes samme
        # ordforråd til å transformere testdataene. Hvis vi hadde
        # latt den se testdataene også, ville modellen fått et
        # urettferdig fortrinn (data leakage).
        vectorizer = TfidfVectorizer(
            max_features=20_000,  # maks antall ord vi tar med (begrenser størrelsen)
            min_df=5,             # ordet må forekomme hos minst 5 brukere
            stop_words="english", # fjerner vanlige, lite informative ord ("the", "and", ...)
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        # Vi trener én egen regresjonsmodell PER Big Five-trekk (fem
        # modeller totalt), siden hvert trekk er et eget tall vi
        # skal gjette.
        predictions = np.zeros_like(y_test, dtype=float)
        for i, trait in enumerate(BIG5_COLS):
            # Ridge-regresjon er vanlig lineær regresjon, men med en
            # "straff" mot for store koeffisienter. Det gjør modellen
            # mer robust når vi har veldig mange features (20 000 ord)
            # i forhold til relativt få brukere.
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train[:, i])
            predictions[:, i] = model.predict(X_test)

        fold_result = evaluate(y_test, predictions, BIG5_COLS)
        fold_result["fold"] = fold_num
        fold_results.append(fold_result)

        # Den "dumme" referansen: gjett alltid gjennomsnittet fra
        # treningssettet, uansett hva brukeren har skrevet.
        mean_pred = np.tile(y_train.mean(axis=0), (len(y_test), 1))
        baseline_result = evaluate(y_test, mean_pred, BIG5_COLS)
        baseline_result["fold"] = fold_num
        baseline_results.append(baseline_result)

    return pd.concat(fold_results), pd.concat(baseline_results)


def summarize(all_folds, label):
    """
    Slår sammen resultatene fra alle foldene til ett gjennomsnitt og
    ett standardavvik per trekk, slik at vi får ett samlet tall å
    rapportere -- i stedet for fem spredte tall fra hver sin fold.
    """
    summary = (
        all_folds.groupby("trait")[["MAE", "Pearson_r"]]
        .agg(["mean", "std"])
    )
    print(f"\n--- {label} (gjennomsnitt ± std over {N_FOLDS} folder) ---")
    print(summary.to_string())
    return summary


def main():
    data = load_and_aggregate()

    model_folds, baseline_folds = run_cross_validation(data)

    summarize(model_folds, "TF-IDF + Ridge-regresjon (tekstmodellen)")
    summarize(baseline_folds, "Baseline: gjett alltid gjennomsnittet")


if __name__ == "__main__":
    main()