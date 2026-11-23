import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMovieRecommender:
    def __init__(self, csv_path: str = "290_movies.csv", model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer(model_name)
        self._build_movie_embeddings()

    def _build_movie_embeddings(self):
        """Génère des embeddings distincts par catégorie sémantique pour chaque film."""
        # 1. Représentation globale / Synopsis + Mots-clés
        synopsis_texts = (
            self.df["Synopsis"].fillna("") + " " + self.df["Keywords"].fillna("")
        ).tolist()

        # 2. Genres et Thèmes
        themes_texts = (
            self.df["Genre"].fillna("") + " - " + self.df["Themes"].fillna("")
        ).tolist()

        # 3. Humeur et atmosphère émotionnelle
        mood_texts = self.df["Mood"].fillna("").tolist()

        # 4. Style narratif et période / cadre
        style_period_texts = (
            self.df["NarrativeStyle"].fillna("") + " | Période : " + self.df["Period"].fillna("")
        ).tolist()

        self.emb_synopsis = self.model.encode(synopsis_texts, normalize_embeddings=True)
        self.emb_themes = self.model.encode(themes_texts, normalize_embeddings=True)
        self.emb_mood = self.model.encode(mood_texts, normalize_embeddings=True)
        self.emb_style = self.model.encode(style_period_texts, normalize_embeddings=True)

    def recommend(
        self,
        free_text: str,
        genres_themes: str = "",
        mood: str = "",
        style_period: str = "",
        top_k: int = 3
    ) -> list[dict]:
        """
        Calcule les similarités par catégorie, agrège les scores pondérés
        et retourne les top_k films avec une explication détaillée.
        """
        # Pondération dynamique : on ne compte que les champs renseignés par l'utilisateur
        weights = {
            "synopsis": 0.45 if free_text.strip() else 0.0,
            "themes": 0.25 if genres_themes.strip() else 0.0,
            "mood": 0.15 if mood.strip() else 0.0,
            "style": 0.15 if style_period.strip() else 0.0,
        }

        total_weight = sum(weights.values())
        if total_weight == 0:
            return []

        # Normalisation des poids pour que leur somme fasse 1.0
        for k in weights:
            weights[k] /= total_weight

        n_movies = len(self.df)
        scores_synopsis = np.zeros(n_movies)
        scores_themes = np.zeros(n_movies)
        scores_mood = np.zeros(n_movies)
        scores_style = np.zeros(n_movies)

        if weights["synopsis"] > 0:
            q_emb = self.model.encode([free_text], normalize_embeddings=True)
            scores_synopsis = cosine_similarity(q_emb, self.emb_synopsis)[0]

        if weights["themes"] > 0:
            q_emb = self.model.encode([genres_themes], normalize_embeddings=True)
            scores_themes = cosine_similarity(q_emb, self.emb_themes)[0]

        if weights["mood"] > 0:
            q_emb = self.model.encode([mood], normalize_embeddings=True)
            scores_mood = cosine_similarity(q_emb, self.emb_mood)[0]

        if weights["style"] > 0:
            q_emb = self.model.encode([style_period], normalize_embeddings=True)
            scores_style = cosine_similarity(q_emb, self.emb_style)[0]

        # Score agrégé final
        final_scores = (
            weights["synopsis"] * scores_synopsis
            + weights["themes"] * scores_themes
            + weights["mood"] * scores_mood
            + weights["style"] * scores_style
        )

        # Classement décroissant
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            breakdown = {}
            if weights["synopsis"] > 0:
                breakdown["Description & Synopsis"] = float(max(0.0, scores_synopsis[idx]))
            if weights["themes"] > 0:
                breakdown["Genres & Thèmes"] = float(max(0.0, scores_themes[idx]))
            if weights["mood"] > 0:
                breakdown["Humeur & Atmosphère"] = float(max(0.0, scores_mood[idx]))
            if weights["style"] > 0:
                breakdown["Style narratif & Cadre"] = float(max(0.0, scores_style[idx]))

            # Identification du critère le plus fort pour l'explication textuelle
            best_axis = max(breakdown, key=breakdown.get) if breakdown else "Global"

            results.append({
                "MovieID": int(row["MovieID"]),
                "Title": row["Title"],
                "Category": row["Category"],
                "Genre": row["Genre"],
                "Synopsis": row["Synopsis"],
                "Themes": row["Themes"],
                "Mood": row["Mood"],
                "NarrativeStyle": row["NarrativeStyle"],
                "Period": row["Period"],
                "Keywords": row["Keywords"],
                "FinalScore": float(max(0.0, final_scores[idx])),
                "Breakdown": breakdown,
                "BestAxis": best_axis,
            })
    def recommend(
        self,
        free_text: str,
        genres_themes: str = "",
        mood: str = "",
        style_period: str = "",
        top_k: int = 3,
        max_movies: int = None
    ) -> list[dict]:
        """
        Calcule les similarités par catégorie sur les `max_movies` premiers films,
        agrège les scores pondérés et retourne les top_k films avec explication.
        """
        # Détermination du nombre de films à analyser
        n_movies = len(self.df) if max_movies is None else min(max_movies, len(self.df))

        weights = {
            "synopsis": 0.45 if free_text.strip() else 0.0,
            "themes": 0.25 if genres_themes.strip() else 0.0,
            "mood": 0.15 if mood.strip() else 0.0,
            "style": 0.15 if style_period.strip() else 0.0,
        }

        total_weight = sum(weights.values())
        if total_weight == 0:
            return []

        for k in weights:
            weights[k] /= total_weight

        scores_synopsis = np.zeros(n_movies)
        scores_themes = np.zeros(n_movies)
        scores_mood = np.zeros(n_movies)
        scores_style = np.zeros(n_movies)

        # On découpe les embeddings avec [:n_movies] (instantané, sans recalcul)
        if weights["synopsis"] > 0:
            q_emb = self.model.encode([free_text], normalize_embeddings=True)
            scores_synopsis = cosine_similarity(q_emb, self.emb_synopsis[:n_movies])[0]

        if weights["themes"] > 0:
            q_emb = self.model.encode([genres_themes], normalize_embeddings=True)
            scores_themes = cosine_similarity(q_emb, self.emb_themes[:n_movies])[0]

        if weights["mood"] > 0:
            q_emb = self.model.encode([mood], normalize_embeddings=True)
            scores_mood = cosine_similarity(q_emb, self.emb_mood[:n_movies])[0]

        if weights["style"] > 0:
            q_emb = self.model.encode([style_period], normalize_embeddings=True)
            scores_style = cosine_similarity(q_emb, self.emb_style[:n_movies])[0]

        final_scores = (
            weights["synopsis"] * scores_synopsis
            + weights["themes"] * scores_themes
            + weights["mood"] * scores_mood
            + weights["style"] * scores_style
        )

        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            breakdown = {}
            if weights["synopsis"] > 0:
                breakdown["Description & Synopsis"] = float(max(0.0, scores_synopsis[idx]))
            if weights["themes"] > 0:
                breakdown["Genres & Thèmes"] = float(max(0.0, scores_themes[idx]))
            if weights["mood"] > 0:
                breakdown["Humeur & Atmosphère"] = float(max(0.0, scores_mood[idx]))
            if weights["style"] > 0:
                breakdown["Style narratif & Cadre"] = float(max(0.0, scores_style[idx]))

            best_axis = max(breakdown, key=breakdown.get) if breakdown else "Global"

            results.append({
                "MovieID": int(row["MovieID"]),
                "Title": row["Title"],
                "Category": row["Category"],
                "Genre": row["Genre"],
                "Synopsis": row["Synopsis"],
                "Themes": row["Themes"],
                "Mood": row["Mood"],
                "NarrativeStyle": row["NarrativeStyle"],
                "Period": row["Period"],
                "Keywords": row["Keywords"],
                "FinalScore": float(max(0.0, final_scores[idx])),
                "Breakdown": breakdown,
                "BestAxis": best_axis,
            })
            
        return results
