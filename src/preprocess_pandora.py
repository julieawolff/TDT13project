"""
Steg 1: filtrer ned PANDORA-dataene til kun brukerne med Big Five-skårer.

Bruk: legg filene dine i samme mappe som dette skriptet
(author_profiles.csv og all_comments_since_2015.csv), og kjør:

    python preprocessing.py

Resultat: en ny, mye mindre fil "filtered_comments.csv" (kun
kommentarer fra de ~1608 brukerne med Big Five-skårer, på engelsk)
som du kan jobbe videre med, i stedet for å røre den store
originalfilen igjen.
"""

import pandas as pd

BIG5_COLS = ["agreeableness", "openness", "conscientiousness",
             "extraversion", "neuroticism"]


def filter_comments(profiles_path, comments_path, output_path,
                     english_only=True, chunksize=200_000):
    # 1. Les brukerprofilene (liten fil, trygt å lese i ett steg)
    profiles = pd.read_csv(profiles_path)
    print(f"Fant {len(profiles)} brukerprofiler totalt.")

    # Behold kun brukere som har alle fem Big Five-skårer utfylt
    has_all_traits = profiles[BIG5_COLS].notna().all(axis=1)
    big5_profiles = profiles.loc[has_all_traits, ["author"] + BIG5_COLS].copy()
    big5_users = set(big5_profiles["author"])
    print(f"{len(big5_users)} brukere har alle fem Big Five-skårer.")

    # Lagre denne lille, rensede profiltabellen for seg selv også –
    # du trenger den senere for å koble på fasit-verdiene (y)
    big5_profiles.to_csv("big5_profiles_filtered.csv", index=False)
    print("Lagret big5_profiles_filtered.csv (bare disse brukerne + skårene)")

    # 2. Les den store kommentarfilen i biter (chunks), behold bare
    #    rader fra de 1608 brukerne, skriv fortløpende til en ny fil
    first_chunk = True
    total_kept = 0
    total_seen = 0

    keep_cols = ["author", "subreddit", "body", "created_utc",
                 "word_count", "lang"]

    for i, chunk in enumerate(pd.read_csv(comments_path, chunksize=chunksize,
                                           low_memory=False)):
        total_seen += len(chunk)

        mask = chunk["author"].isin(big5_users)
        if english_only and "lang" in chunk.columns:
            mask &= (chunk["lang"] == "en")

        filtered = chunk.loc[mask, [c for c in keep_cols if c in chunk.columns]]

        filtered.to_csv(output_path, mode="w" if first_chunk else "a",
                         header=first_chunk, index=False)
        first_chunk = False
        total_kept += len(filtered)

        if i % 5 == 0:
            print(f"  ...lest {total_seen:,} rader totalt, "
                  f"beholdt {total_kept:,} hittil")

    print(f"\nFerdig. {total_kept:,} av {total_seen:,} kommentarer "
          f"skrevet til {output_path}")


if __name__ == "__main__":
    filter_comments(
        profiles_path="author_profiles.csv",
        comments_path="all_comments_since_2015.csv",
        output_path="filtered_comments.csv",
        english_only=True,
    )