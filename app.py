import html
import pandas as pd
import numpy as np
import joblib
import scipy.sparse
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Audible Insights", page_icon="🎧", layout="wide")

ORANGE = "#F28C28"


# ---------------- Custom styling ----------------
st.markdown("""
<style>
.block-container {padding-top: 3rem; max-width: 1200px;}

/* top banner */
.hero {
    background: linear-gradient(120deg, #2B2118 0%, #5C3A1E 60%, #F28C28 140%);
    color: white; padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.5rem;
}
.hero h1 {color: white; font-size: 2.1rem; margin: 0 0 .4rem 0;}
.hero p {color: #F5E6D3; font-size: 1.05rem; margin: 0;}

/* stat boxes */
.stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 1.5rem;}
.stat {background: white; border: 1px solid #F1E2CC; border-radius: 14px; padding: 1rem; text-align: center;}
.stat .num {font-size: 1.7rem; font-weight: 700; color: #F28C28;}
.stat .label {font-size: .85rem; color: #7A6A5A;}

/* book cards in a responsive grid */
.book-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; margin: .5rem 0 1.5rem 0;}
.book-card {
    background: white; border: 1px solid #F1E2CC; border-radius: 16px; padding: 1rem;
    display: flex; gap: 14px; transition: transform .15s, box-shadow .15s;
}
.book-card:hover {transform: translateY(-3px); box-shadow: 0 8px 22px rgba(242, 140, 40, .18);}
.cover {
    min-width: 72px; width: 72px; height: 100px; border-radius: 8px; color: white;
    display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.4rem;
}
.book-info {flex: 1; min-width: 0;}
.book-title {font-weight: 700; font-size: 1rem; line-height: 1.3; margin-bottom: 2px; color: #2B2118;}
.book-author {font-size: .85rem; color: #7A6A5A; margin-bottom: 6px;}
.stars {color: #F28C28; font-size: .9rem;}
.meta {font-size: .8rem; color: #7A6A5A;}
.desc {font-size: .82rem; color: #4A3F35; margin-top: 6px; line-height: 1.4;}
.tag {
    display: inline-block; background: #FFF0DB; color: #A65C12; border-radius: 20px;
    padding: 2px 10px; font-size: .72rem; margin: 4px 4px 0 0;
}
.match {float: right; background: #2B2118; color: white; border-radius: 20px; padding: 2px 9px; font-size: .72rem;}

/* selected book banner */
.selected {background: #FFF0DB; border-left: 5px solid #F28C28; border-radius: 10px; padding: .8rem 1rem; margin: .5rem 0 1rem 0;}

section[data-testid="stSidebar"] {background: #2B2118;}
section[data-testid="stSidebar"] * {color: #F5E6D3;}

@media (max-width: 640px) {
    .hero {padding: 1.3rem;}
    .hero h1 {font-size: 1.5rem;}
    .book-grid {grid-template-columns: 1fr;}
}
</style>
""", unsafe_allow_html=True)


# ---------------- Load data and models ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/books_with_clusters.csv")
    df["Genres"] = df["Genres"].fillna("")
    df["Description"] = df["Description"].fillna("")
    return df

@st.cache_resource
def load_models():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    tfidf_matrix = scipy.sparse.load_npz("models/tfidf_matrix.npz")
    return tfidf, tfidf_matrix

df = load_data()
tfidf, tfidf_matrix = load_models()


# ---------------- Recommendation functions ----------------
def content_recommend(book_index, n):
    similarity = cosine_similarity(tfidf_matrix[book_index], tfidf_matrix)[0]
    similarity[book_index] = -1
    top_index = similarity.argsort()[::-1][:n]
    return df.loc[top_index], similarity[top_index]

def cluster_recommend(book_index, n):
    cluster = df.loc[book_index, "Cluster"]
    same_cluster = df[(df["Cluster"] == cluster) & (df.index != book_index)]
    result = same_cluster.sort_values(by="Score", ascending=False).head(n)
    similarity = cosine_similarity(tfidf_matrix[book_index], tfidf_matrix[result.index])[0]
    return result, similarity

def hybrid_recommend(book_index, n):
    similarity = cosine_similarity(tfidf_matrix[book_index], tfidf_matrix)[0]
    same_cluster = (df["Cluster"] == df.loc[book_index, "Cluster"]).astype(int).values
    final_score = 0.7 * similarity + 0.2 * df["Score"].values + 0.1 * same_cluster
    final_score[book_index] = -1
    top_index = final_score.argsort()[::-1][:n]
    return df.loc[top_index], similarity[top_index]

def keyword_recommend(text, n):
    text_vector = tfidf.transform([text.lower()])
    similarity = cosine_similarity(text_vector, tfidf_matrix)[0]
    final_score = 0.8 * similarity + 0.2 * df["Score"].values
    top_index = final_score.argsort()[::-1][:n]
    return df.loc[top_index], similarity[top_index]

@st.cache_data
def get_all_genres():
    genre_list = []
    for g in df["Genres"]:
        if g != "":
            genre_list.extend(g.split("|"))
    return pd.Series(genre_list).value_counts()


# ---------------- Book cards ----------------
cover_colors = ["#F28C28", "#5C3A1E", "#C0582B", "#2E6F73", "#7B4B94", "#3B5BA5", "#A65C12", "#4F7942"]

def make_stars(rating):
    if pd.isnull(rating):
        return "<span class='meta'>Not rated yet</span>"
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full) + f" <b>{rating}</b>"

def book_card(row, match=None):
    title = html.escape(str(row["Book Name"]))
    author = html.escape(str(row["Author"]))
    initials = "".join([w[0] for w in str(row["Book Name"]).split()[:2] if w[0].isalnum()]).upper() or "📖"
    color = cover_colors[len(str(row["Book Name"])) % len(cover_colors)]

    tags = ""
    if row["Genres"] != "":
        for g in row["Genres"].split("|")[:2]:
            tags += f"<span class='tag'>{html.escape(g)}</span>"

    desc = ""
    if row["Description"] != "":
        desc = f"<div class='desc'>{html.escape(row['Description'][:140])}...</div>"

    match_badge = f"<span class='match'>{int(match * 100)}% match</span>" if match is not None and match > 0 else ""

    return (f"<div class='book-card'>"
            f"<div class='cover' style='background:{color}'>{initials}</div>"
            f"<div class='book-info'>{match_badge}"
            f"<div class='book-title'>{title}</div>"
            f"<div class='book-author'>by {author}</div>"
            f"<div class='stars'>{make_stars(row['Rating'])}</div>"
            f"<div class='meta'>{int(row['Number of Reviews']):,} reviews · ₹{int(row['Price'])}</div>"
            f"{tags}{desc}</div></div>")

def show_books(books, matches=None):
    cards = ""
    for i, (index, row) in enumerate(books.iterrows()):
        match = matches[i] if matches is not None else None
        cards += book_card(row, match)
    st.markdown(f"<div class='book-grid'>{cards}</div>", unsafe_allow_html=True)

def stat_boxes(items):
    boxes = ""
    for label, value in items:
        boxes += f"<div class='stat'><div class='num'>{value}</div><div class='label'>{label}</div></div>"
    st.markdown(f"<div class='stats'>{boxes}</div>", unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)

def style_chart(fig):
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10), font=dict(color="#2B2118"))
    return fig


# ---------------- Sidebar ----------------
st.sidebar.markdown("## 🎧 Audible Insights")
st.sidebar.caption("Intelligent book recommendations")
page = st.sidebar.radio("Menu", ["🏠 Home", "📖 Similar Books", "💡 Search by Interest",
                                 "🎭 Browse by Genre", "💎 Hidden Gems", "📊 Insights"],
                        label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"📚 {len(df):,} books · ✍️ {df['Author'].nunique():,} authors")
st.sidebar.caption("Built with TF-IDF, K-Means and a Hybrid recommender")


# ---------------- Home ----------------
if page == "🏠 Home":
    hero("Find your next great listen 🎧", "Personalised audiobook recommendations powered by NLP and machine learning.")

    stat_boxes([("Books", f"{len(df):,}"),
                ("Authors", f"{df['Author'].nunique():,}"),
                ("Average Rating", f"⭐ {df['Rating'].mean():.2f}"),
                ("Book Clusters", df["Cluster"].nunique())])

    st.subheader("🔥 Trending now")
    st.caption("Most popular books based on rating and number of reviews")
    show_books(df.sort_values(by="Score", ascending=False).head(6))

    st.subheader("✨ What would you like to do?")
    col1, col2, col3 = st.columns(3)
    col1.info("**📖 Similar Books**\n\nLoved a book? Find more like it.")
    col2.info("**🎭 Browse by Genre**\n\nNew here? Start with your favourite genres.")
    col3.info("**💎 Hidden Gems**\n\nGreat books that few people know about.")


# ---------------- Similar Books ----------------
elif page == "📖 Similar Books":
    hero("📖 Books like the one you loved", "Pick a book and we'll find similar ones.")

    book_names = df.sort_values(by="Number of Reviews", ascending=False)["Book Name"].unique()
    selected_book = st.selectbox("🔎 Search for a book", book_names,
                                 help="Start typing a title. Popular books are shown first.")

    col1, col2 = st.columns([2, 1])
    model = col1.segmented_control("Recommendation model", ["Hybrid", "Content-Based", "Cluster-Based"],
                                   default="Hybrid") or "Hybrid"
    n = col2.slider("How many books?", 3, 12, 6)

    book_index = df[df["Book Name"] == selected_book].sort_values(by="Number of Reviews", ascending=False).index[0]
    book = df.loc[book_index]
    st.markdown(f"<div class='selected'><b>You picked:</b> {html.escape(book['Book Name'])} "
                f"by {html.escape(book['Author'])} · {html.escape(book['Primary Genre'])}</div>",
                unsafe_allow_html=True)

    model_help = {
        "Hybrid": "⭐ Best model: 70% text similarity + 20% popularity + 10% same cluster",
        "Content-Based": "Books with the most similar title, genres and description (TF-IDF + cosine similarity)",
        "Cluster-Based": "The most popular books from the same K-Means cluster",
    }
    st.caption(model_help[model])

    if model == "Content-Based":
        result, matches = content_recommend(book_index, n)
    elif model == "Cluster-Based":
        result, matches = cluster_recommend(book_index, n)
    else:
        result, matches = hybrid_recommend(book_index, n)

    st.subheader("Recommended for you")
    show_books(result, matches)


# ---------------- Search by Interest ----------------
elif page == "💡 Search by Interest":
    hero("💡 Tell us what you're in the mood for", "Describe a topic, theme or feeling, and we'll match books to it.")

    text = st.text_input("What do you want to listen to?", placeholder="e.g. habits productivity success")

    st.caption("Try one of these:")
    examples = ["money investing", "murder mystery thriller", "meditation mindfulness", "space science fiction", "leadership business"]
    cols = st.columns(len(examples))
    for i in range(len(examples)):
        if cols[i].button(examples[i], width="stretch"):
            st.session_state["search_text"] = examples[i]
    if text == "" and "search_text" in st.session_state:
        text = st.session_state["search_text"]

    n = st.slider("How many books?", 3, 12, 6)

    if text.strip() != "":
        result, matches = keyword_recommend(text, n)
        if matches.max() == 0:
            st.warning("No books match these words. Try different words.")
        else:
            st.subheader(f"Results for “{text}”")
            show_books(result, matches)
    else:
        st.info("Type something above or click an example to get started.")


# ---------------- Browse by Genre ----------------
elif page == "🎭 Browse by Genre":
    hero("🎭 Explore by genre", "New listener? Pick the genres you enjoy and see the best-rated books.")

    genre_counts = get_all_genres()
    col1, col2 = st.columns([3, 1])
    genres = col1.multiselect("Choose your favourite genres", genre_counts.index, default=["Science Fiction"])
    min_rating = col2.slider("Minimum rating", 3.0, 5.0, 4.0, 0.1)

    st.caption("Popular genres:")
    st.markdown(" ".join([f"<span class='tag'>{html.escape(g)}</span>" for g in genre_counts.head(10).index]),
                unsafe_allow_html=True)

    if len(genres) == 0:
        st.info("Choose at least one genre.")
    else:
        mask = df["Genres"].apply(lambda g: any(genre in g.split("|") for genre in genres))
        result = df[mask & (df["Rating"] >= min_rating)].sort_values(by="Score", ascending=False).head(9)
        if len(result) == 0:
            st.warning("No books found. Try a lower minimum rating.")
        else:
            st.subheader(f"Top picks in {', '.join(genres)}")
            show_books(result)


# ---------------- Hidden Gems ----------------
elif page == "💎 Hidden Gems":
    hero("💎 Hidden gems", "Highly rated books that haven't been discovered by many listeners yet.")

    col1, col2, col3 = st.columns(3)
    min_rating = col1.slider("Minimum rating", 4.5, 5.0, 4.7, 0.1)
    max_reviews = col2.slider("Maximum reviews", 10, 200, 50)
    genre_options = ["All genres"] + list(get_all_genres().head(30).index)
    genre = col3.selectbox("Genre", genre_options)

    gems = df[(df["Rating"] >= min_rating) & (df["Number of Reviews"] >= 5) & (df["Number of Reviews"] <= max_reviews)]
    if genre != "All genres":
        gems = gems[gems["Genres"].str.contains(genre, regex=False)]
    gems = gems.sort_values(by=["Rating", "Number of Reviews"], ascending=False)

    st.success(f"Found **{len(gems)}** hidden gems. Showing the top 12.")
    show_books(gems.head(12))


# ---------------- Insights ----------------
elif page == "📊 Insights":
    hero("📊 Data insights", "What the Audible catalog tells us about genres, authors and ratings.")
    rated = df[df["Rating"].notnull()]

    tab1, tab2, tab3, tab4 = st.tabs(["🎭 Genres", "✍️ Authors", "⭐ Ratings", "🤖 Model Results"])

    with tab1:
        top_genres = get_all_genres().head(15).sort_values()
        fig = px.bar(x=top_genres.values, y=top_genres.index, orientation="h",
                     title="Top 15 Genres", labels={"x": "Number of books", "y": ""},
                     color_discrete_sequence=[ORANGE])
        st.plotly_chart(style_chart(fig), width="stretch")

    with tab2:
        author_stats = rated.groupby("Author").agg(books=("Rating", "count"), avg_rating=("Rating", "mean"))
        author_stats = author_stats[author_stats["books"] >= 3].sort_values(by="avg_rating", ascending=False).head(10)
        author_stats = author_stats.sort_values(by="avg_rating")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(author_stats, x="avg_rating", y=author_stats.index, orientation="h",
                         title="Highest Rated Authors (min 3 books)", range_x=[4.5, 5],
                         labels={"avg_rating": "Average rating", "y": ""}, color_discrete_sequence=[ORANGE])
            st.plotly_chart(style_chart(fig), width="stretch")
        with col2:
            top_authors = df["Author"].value_counts().head(10).sort_values()
            fig = px.bar(x=top_authors.values, y=top_authors.index, orientation="h",
                         title="Authors with Most Books", labels={"x": "Books", "y": ""},
                         color_discrete_sequence=["#5C3A1E"])
            st.plotly_chart(style_chart(fig), width="stretch")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(rated, x="Rating", nbins=20, title="Rating Distribution",
                               color_discrete_sequence=[ORANGE])
            st.plotly_chart(style_chart(fig), width="stretch")
        with col2:
            fig = px.scatter(rated, x="Number of Reviews", y="Rating", log_x=True, opacity=0.4,
                             hover_name="Book Name", title="Rating vs Number of Reviews",
                             color_discrete_sequence=["#5C3A1E"])
            st.plotly_chart(style_chart(fig), width="stretch")

        corr = df[["Rating", "Number of Reviews", "Price", "Listening Minutes"]].corr(method="spearman").round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="Oranges", title="Correlation Heatmap")
        st.plotly_chart(style_chart(fig), width="stretch")

        st.info("💡 Books with few reviews have very mixed ratings. With more reviews, ratings settle around 4.5.")

    with tab4:
        st.write("Evaluated on 300 test books. A recommendation counts as relevant if it shares a genre with the selected book.")
        results = pd.DataFrame({
            "Model": ["Content-Based", "Cluster-Based", "Hybrid"],
            "Precision@10": [0.392, 0.194, 0.434],
            "Recall@10": [0.120, 0.027, 0.109],
            "RMSE": [0.347, 0.362, 0.347],
        })
        col1, col2 = st.columns([1, 1])
        col1.dataframe(results, hide_index=True, width="stretch")
        fig = px.bar(results, x="Model", y="Precision@10", title="Precision@10 by Model", text="Precision@10",
                     color="Model", color_discrete_sequence=["#C0582B", "#5C3A1E", ORANGE])
        fig.update_layout(showlegend=False)
        col2.plotly_chart(style_chart(fig), width="stretch")
        st.success("🏆 The Hybrid model gives the best precision, so it's the default in the app.")
