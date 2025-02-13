import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Load corpus from a CSV file
def load_corpus(document_path):
    intent_questions = {}
    intent_answers = {}

    try:
        df = pd.read_csv(document_path, encoding='utf-8')
        
        for _, row in df.iterrows():
            # Extract columns
            question = row.get('Question')
            answer = row.get('Answer')
            intent = row.get('Intent')

          
            #ensure strings are properly stripped
            if pd.notnull(question):
                question = question.strip()
            else:
                question = ""

            if pd.notnull(answer):
                answer = answer.strip()
            else:
                answer = ""
            
            # Initialize dictionaries if new intent
            if intent not in intent_questions:
                intent_questions[intent] = []
            if intent not in intent_answers:
                intent_answers[intent] = set()  # use set to ensure uniqueness

            # Add data
            if question:
                intent_questions[intent].append(question)
            if answer:
                intent_answers[intent].add(answer)
                
    except Exception as e:
        print(f"Error reading file {document_path}: {e}")

    #combine questions and answers
    corpus = {}
    answers = {}
    for intent in intent_questions:
        combined_questions = " ".join(intent_questions[intent])
        
        combined_answers = " ".join(intent_answers[intent])

        corpus[intent] = combined_questions
        answers[intent] = combined_answers

    return corpus, answers


def preprocess_small_talkQA(corpus):
    documents = list(corpus.values())
    tfidf_vect = TfidfVectorizer()
    tfidf_matrix = tfidf_vect.fit_transform(documents)
    return tfidf_matrix, tfidf_vect

# Update the fallback small talk search with TF-IDF
def search_document(corpus, answers, tfidf_matrix, tfidf_vect, user_query):
    threshold=0.5
    query_vector = tfidf_vect.transform([user_query])  # Transform the user query
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()  # Compute similarity scores
    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[best_match_index]

    # Return the corresponding response if similarity exceeds the threshold
    if best_score >= threshold:
        best_intent = list(corpus.keys())[best_match_index]
        if best_intent == "smalltalk_user_name":
            return best_intent
        return answers[best_intent]
    return None
