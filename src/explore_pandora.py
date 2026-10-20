"""
Steg 3: utforsk de filtrerte PANDORA-dataene.

Kjør etter at du har laget filtered_comments.csv og
big5_profiles_filtered.csv med preprocess_pandora.py:

    python explore_pandora.py
"""

import pandas as pd

# Subreddits som direkte avslører personlighetstype -- disse bør IKKE
# brukes som nettverksfeature, siden de ville "lekke" fasiten (label
# leakage). Utvid listen hvis du finner flere slike i dine data.
PERSONALITY_SUBREDDITS = {
    "mbti", "intp", "infj", "infp", "intj", "entp", "enfp", "enfj", "entj",
    "istj", "isfj", "istp", "isfp", "estj", "esfj", "estp", "esfp",
    "enneagram", "myersbriggs", "personalitycafe", "MBTIPlus",
}


def explore(comments_path="filtered_comments.csv",
            profiles_path="big5_profiles_filtered.csv"):

    comments = pd.read_csv(comments_path)
    profiles = pd.read_csv(profiles_path)

    print("=" * 60)
    print("GRUNNTALL")
    print("=" * 60)
    print(f"Antall kommentarer: {len(comments):,}")
    print(f"Antall brukere med Big Five-skårer: {len(profiles)}")
    print(f"Antall brukere som faktisk har kommentarer i dataen: "
          f"{comments['author'].nunique()}")

    print("\n" + "=" * 60)
    print("KOMMENTARER PER BRUKER")
    print("=" * 60)
    per_user = comments.groupby("author").size()
    print(per_user.describe())
    print(f"\nBrukere med færre enn 20 kommentarer: "
          f"{(per_user < 20).sum()} ({(per_user < 20).mean():.1%})")

    print("\n" + "=" * 60)
    print("LENGDE PÅ KOMMENTARER (word_count)")
    print("=" * 60)
    if "word_count" in comments.columns:
        print(comments["word_count"].describe())
        print(f"Kommentarer med færre enn 5 ord: "
              f"{(comments['word_count'] < 5).sum():,}")

    print("\n" + "=" * 60)
    print("SUBREDDITS")
    print("=" * 60)
    print(f"Antall unike subreddits totalt: {comments['subreddit'].nunique()}")
    sub_counts = comments["subreddit"].value_counts()
    print("\nTopp 15 mest aktive subreddits:")
    print(sub_counts.head(15))

    found_personality_subs = set(comments["subreddit"].str.lower()) & PERSONALITY_SUBREDDITS
    n_leaky_comments = comments["subreddit"].str.lower().isin(PERSONALITY_SUBREDDITS).sum()
    print(f"\n⚠️  Personlighets-relaterte subreddits funnet i dataene: "
          f"{sorted(found_personality_subs)}")
    print(f"⚠️  Antall kommentarer i disse (bør trolig filtreres bort fra "
          f"nettverksfeatureet): {n_leaky_comments:,} "
          f"({n_leaky_comments / len(comments):.1%} av alle kommentarer)")

    print("\n" + "=" * 60)
    print("MANGLENDE VERDIER OG DUPLIKATER")
    print("=" * 60)
    print("Manglende verdier per kolonne (comments):")
    print(comments.isna().sum())
    n_deleted = comments["body"].isin(["[deleted]", "[removed]"]).sum()
    print(f"\nSlettede/fjernede kommentarer: {n_deleted:,}")
    n_dupes = comments.duplicated(subset=["author", "body"]).sum()
    print(f"Duplikate (bruker, tekst)-par: {n_dupes:,}")

    print("\n" + "=" * 60)
    print("BIG FIVE-SKÅRER")
    print("=" * 60)
    big5_cols = ["agreeableness", "openness", "conscientiousness",
                 "extraversion", "neuroticism"]
    print(profiles[big5_cols].describe())

    print("\n" + "=" * 60)
    print("TIDSSPENN")
    print("=" * 60)
    if "created_utc" in comments.columns:
        dates = pd.to_datetime(comments["created_utc"], unit="s")
        print(f"Eldste kommentar: {dates.min()}")
        print(f"Nyeste kommentar: {dates.max()}")


if __name__ == "__main__":
    explore()