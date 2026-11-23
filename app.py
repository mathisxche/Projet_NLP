import streamlit as st
from recommender import SemanticMovieRecommender

st.set_page_config(
    page_title="CinéSense NLP - Recommandation Sémantique",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    return SemanticMovieRecommender("290_movies.csv")

recommender = load_recommender()

# --- EN-TÊTE ---
st.title("🎬 CinéSense NLP : Recommandation Sémantique de Films")
st.markdown(
    "Décrivez le film dont vous avez envie en **langage naturel**. "
    "Notre moteur NLP analyse le sens profond de vos attentes pour vous proposer les **3 films les plus pertinents** avec une explication détaillée."
)

st.divider()

# --- FORMULAIRE UTILISATEUR ---
with st.form("user_preferences_form"):
    st.subheader("1. Décrivez votre envie en texte libre")
    free_text = st.text_area(
        "Quel genre d'histoire ou d'expérience recherchez-vous ?",
        placeholder="Ex : J'aimerais regarder un film de science-fiction émouvant qui explore les relations humaines, l'intelligence artificielle et l'avenir de la société.",
        height=100
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2. Genres & Thèmes")
        selected_genres = st.multiselect(
            "Genres préférés",
            ["Science-Fiction", "Drame", "Thriller", "Romance", "Comédie",
             "Action", "Aventure", "Animation", "Historique", "Fantasy", "Policier", "Biopic", "Horreur"]
        )
        custom_themes = st.text_input(
            "Thèmes spécifiques (optionnel)",
            placeholder="Ex : intelligence artificielle, amitié, vengeance, écologie, famille..."
        )

        st.subheader("3. Humeur & Atmosphère")
        selected_moods = st.multiselect(
            "Quelle ambiance émotionnelle recherchez-vous ?",
            ["Émouvant", "Tendu", "Sombre", "Joyeux", "Poétique", "Mélancolique",
             "Épique", "Inspirant", "Drôle", "Mystérieux", "Contemplatif", "Anxiogène"]
        )

    with col2:
        st.subheader("4. Période & Cadre")
        period_choice = st.selectbox(
            "Période historique ou cadre préféré",
            [
                "Peu importe",
                "Contemporain",
                "Futur proche / Futur dystopique",
                "Futur lointain / Espace",
                "Années 1980 - 1990",
                "Années 1940 - 1960",
                "Seconde / Première Guerre mondiale",
                "Antiquité / XVIIIe - XIXe siècle",
                "Médiéval fantastique / Monde imaginaire"
            ]
        )

        st.subheader("5. Questions guidées sur le style narratif")
        pace = st.radio(
            "Quel rythme préférez-vous ?",
            ["Peu importe", "Lent et contemplatif", "Équilibré et classique", "Rapide, intense et rythmé"],
            horizontal=True
        )
        narrative_pref = st.multiselect(
            "Préférences narratives",
            [
                "Axé sur les dialogues et les émotions",
                "Scénario complexe / Twist final",
                "Grand spectacle et visuel marquant",
                "Histoire vraie / Biopic",
                "Critique sociale / Satire",
                "Philosophique / Réflexion profonde"
            ]
        )

    submitted = st.form_submit_button("🔍 Trouver mes 3 films recommandés", type="primary", use_container_width=True)

# --- TRAITEMENT ET AFFICHAGE DES RÉSULTATS ---
if submitted:
    # Construction des requêtes textuelles par catégorie
    genres_themes_query = " ".join(selected_genres) + " " + custom_themes
    mood_query = ", ".join(selected_moods)

    style_parts = []
    if period_choice != "Peu importe":
        style_parts.append(f"Période : {period_choice}")
    if pace != "Peu importe":
        style_parts.append(f"Rythme : {pace}")
    if narrative_pref:
        style_parts.append("Style : " + ", ".join(narrative_pref))
    style_period_query = " | ".join(style_parts)

    if not any([free_text.strip(), genres_themes_query.strip(), mood_query.strip(), style_period_query.strip()]):
        st.warning("Veuillez remplir au moins un champ (par exemple la description en texte libre) pour lancer l'analyse.")
    else:
        with st.spinner("Analyse sémantique et calcul des similarités en cours..."):
            recommendations = recommender.recommend(
                free_text=free_text,
                genres_themes=genres_themes_query,
                mood=mood_query,
                style_period=style_period_query,
                top_k=3
            )

        st.divider()
        st.header("🏆 Top 3 des films recommandés pour vous")

        for rank, movie in enumerate(recommendations, start=1):
            score_pct = int(round(movie["FinalScore"] * 100))

            with st.container(border=True):
                head_col, score_col = st.columns([4, 1])
                with head_col:
                    st.subheader(f"#{rank} — {movie['Title']}")
                    st.caption(f"📂 {movie['Category']}  |  🕰️ {movie['Period']}  |  🎭 {movie['Mood']}")
                with score_col:
                    st.metric(label="Affinité globale", value=f"{score_pct}%")

                st.write(f"**Synopsis :** {movie['Synopsis']}")

                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.markdown(f"**Thèmes :** {movie['Themes']}")
                    st.markdown(f"**Style narratif :** {movie['NarrativeStyle']}")
                with info_col2:
                    st.markdown(f"**Mots-clés :** `{movie['Keywords']}`")

                # --- SECTION EXPLICABILITÉ ---
                with st.expander("🔎 Pourquoi ce film vous est-il recommandé ? (Explicabilité NLP)", expanded=(rank == 1)):
                    st.info(
                        f"Ce film se distingue particulièrement sur l'axe **{movie['BestAxis']}**, "
                        f"avec une forte proximité sémantique par rapport à votre requête."
                    )
                    for axis, val in movie["Breakdown"].items():
                        pct = int(round(val * 100))
                        st.write(f"**{axis}** : {pct}%")
                        st.progress(min(max(val, 0.0), 1.0))
