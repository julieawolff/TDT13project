"""
Steg 4: rydd og klargjør dataene til modellering.

Tar inn filtered_comments.csv + big5_profiles_filtered.csv (fra
preprocess_pandora.py) og gjør følgende:

1. Fjerner duplikate (bruker, tekst)-par
2. Fjerner [deleted]/[removed]-kommentarer
3. Fjerner brukere med færre enn MIN_COMMENTS kommentarer
4. Lager en egen "clean" kommentar-tabell for tekstmodellen
5. Lager en egen subreddit-tabell for nettverksfeatureet, der
   personlighets-relaterte subreddits er fjernet (for å unngå
   label leakage), men lagrer også en versjon MED dem i tilfelle
   du vil vise fram forskjellen i rapporten

Kjør:
    python clean_pandora.py
"""

import pandas as pd

MIN_COMMENTS_PER_USER = 20

# Subreddits som direkte avslører personlighetstype -- fjernes fra
# nettverksfeatureet for å unngå label leakage (bekreftet i dataene:
# ~10% av alle kommentarer ligger i disse)
PERSONALITY_SUBREDDITS = {
    "mbti", "intp", "infj", "infp", "intj", "entp", "enfp", "enfj", "entj",
    "istj", "isfj", "istp", "isfp", "estj", "esfj", "estp", "esfp",
    "enneagram", "myersbriggs", "personalitycafe", "MBTIPlus",
}



def clean(comments_path="filtered_comments.csv",
          profiles_path="big5_profiles_filtered.csv"):

    comments = pd.read_csv(comments_path)
    profiles = pd.read_csv(profiles_path)
    print(f"Start: {len(comments):,} kommentarer, {len(profiles)} brukere")

    # 1. Fjern slettede/fjernede kommentarer
    comments = comments[~comments["body"].isin(["[deleted]", "[removed]"])]
    print(f"Etter å fjerne slettede/fjernede: {len(comments):,} kommentarer")

    # 2. Fjern duplikate (bruker, tekst)-par (behold første forekomst)
    before = len(comments)
    comments = comments.drop_duplicates(subset=["author", "body"])
    print(f"Fjernet {before - len(comments):,} duplikater, "
          f"{len(comments):,} kommentarer igjen")

    # 3. Fjern brukere med for få kommentarer
    counts = comments.groupby("author").size()
    keep_users = counts[counts >= MIN_COMMENTS_PER_USER].index
    comments = comments[comments["author"].isin(keep_users)]
    profiles = profiles[profiles["author"].isin(keep_users)]
    print(f"Brukere med minst {MIN_COMMENTS_PER_USER} kommentarer: "
          f"{len(keep_users)}")
    print(f"Kommentarer igjen: {len(comments):,}")

    # 4. Lagre den rensede tekst-tabellen (brukes til tekstmodellen)
    comments.to_csv("comments_clean.csv", index=False)
    profiles.to_csv("profiles_clean.csv", index=False)
    print("\nLagret comments_clean.csv og profiles_clean.csv "
          "(til tekstmodellen)")

    # 5. Lag nettverkstabellen: subreddit-deltakelse PER BRUKER,
    #    med personlighets-subreddits fjernet
    is_leaky = comments["subreddit"].str.lower().isin(PERSONALITY_SUBREDDITS)
    network_comments = comments[~is_leaky]

    subreddit_counts = (
        network_comments
        .groupby(["author", "subreddit"])
        .size()
        .reset_index(name="count")
    )
    subreddit_counts.to_csv("subreddit_counts_clean.csv", index=False)
    print(f"\nLagret subreddit_counts_clean.csv "
          f"({is_leaky.sum():,} kommentarer i personlighets-subreddits "
          f"ble utelatt fra nettverksfeatureet)")
    print(f"Unike subreddits igjen i nettverksfeatureet: "
          f"{network_comments['subreddit'].nunique()}")

    print("\nFerdig! Du kan nå bygge:")
    print("  - tekstmodellen fra comments_clean.csv + profiles_clean.csv")
    print("  - subreddit-vektoren fra subreddit_counts_clean.csv")


if __name__ == "__main__":
    clean()