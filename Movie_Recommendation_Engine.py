from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = {
    "Avatar": "action adventure sci-fi",
    "Titanic": "romance drama",
    "The Avengers": "action superhero sci-fi",
    "Iron Man": "action superhero technology",
    "The Notebook": "romance drama love"
}

vectorizer = CountVectorizer()
movie_vectors = vectorizer.fit_transform(movies.values())

similarity = cosine_similarity(movie_vectors)

movie_names = list(movies.keys())

favorite_movie = "Avatar"
index = movie_names.index(favorite_movie)

scores = list(enumerate(similarity[index]))
scores = sorted(scores, key=lambda x: x[1], reverse=True)

print("Recommended Movies:")
for i, score in scores[1:4]:
    print(movie_names[i])