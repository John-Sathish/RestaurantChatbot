import json
import numpy as np
import random
from smallQA import load_corpus, preprocess_small_talkQA,search_document
from datetime import datetime,timedelta
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import re
from collections import Counter
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

reservation_json = r'./reservations.json'
with open(reservation_json, 'r') as file:
    reservations_data = json.load(file)


intents_df = pd.read_csv(r'./Balanced_Restaurant_Intent.csv')
prompts = intents_df['prompt'].tolist()
intents = intents_df['intent'].tolist()

small_talk_corpus_path = r'./Combined_Small_talk_Dataset.csv'
small_talk_corpus,small_talk_answers = load_corpus(small_talk_corpus_path)
small_talk_matrix, small_talk_vectoriser = preprocess_small_talkQA(small_talk_corpus)

def evaluate_classifier(classifier, X_test, y_test, model_name=""):
    y_pred = classifier.predict(X_test)
    print(f"Performance for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def cross_validate_classifier(classifier, X, y, model_name=""):
    

    # Get the size of the smallest class
    class_counts = Counter(y)
    n_splits = min(class_counts.values()) 
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(classifier, X, y, cv=kfold, scoring='accuracy')

    print(f"Cross-validation for {model_name}:")
    print("Scores:", scores)
    print("Average accuracy:", scores.mean())
    
#Dataset for the senttimental classifier
mood_data = {
    "angry": [
        "I'm upset", 
        "This is frustrating", 
        "I hate this", 
        "You're useless", 
        "Why can't things go right?", 
        "I'm so annoyed", 
        "This makes me furious", 
        "I'm livid right now", 
        "Everything is so irritating", 
        "This is unacceptable"
    ],
    "happy": [
        "I'm so happy", 
        "This is great", 
        "I love this", 
        "You're awesome", 
        "I'm thrilled", 
        "This makes me smile", 
        "I'm on top of the world", 
        "I couldn't be happier", 
        "This is amazing", 
        "Everything is wonderful"
    ],
    "neutral": [
        "Okay", 
        "Sure", 
        "Fine", 
        "Alright", 
        "I don't mind", 
        "It doesn't matter", 
        "That's okay", 
        "No problem", 
        "I'm indifferent", 
        "It's neither here nor there"
    ]
}

mood_sentences = [sentence for mood in mood_data.values() for sentence in mood]
mood_intents = [intent for intent, sentences in mood_data.items() for i in sentences]

#tokenise and remove stopwords
def preprocess_text(text):
   
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text.lower())
    return " ".join([token for token in tokens if token.isalnum() and token not in stop_words])


#checks daate not in the past and not more than 1 year into future
def check_date(date_str):
    try:
        input_date = datetime.strptime(date_str, "%d-%m-%Y")
        current_date = datetime.now()

        
        if input_date < current_date:
            return False, "Chatbot: The date cannot be in the past."
        elif input_date > current_date + timedelta(days=365):
            return False, "Chatbot: The date cannot be more than one year in the future."
        else:
            return True, None  
    except ValueError:
        return False, "ChatBot: Please enter in the format DD-MM-YYYY"

#check the time is in right for format(24hrs or 12hrs format)
def check_time(time_str):
    time = None
    time_pattern = re.search(r"(\b(1[0-2]|0?[1-9]):[0-5][0-9]\s(AM|PM|am|pm)\b)|(\b(2[0-3]|[01]?[0-9]):[0-5][0-9]\b)|(\b(1[0-2]|0?[1-9])\s(AM|PM|am|pm)\b)", 
                             time_str, re.IGNORECASE)

    if time_pattern:
        
        time = time_pattern.group(0) 
    return time

#fuction for personal greeting using current day and time
def personal_greeting():
    current_hour = datetime.now().hour
    current_day = datetime.now().strftime("%A")
    if 5 <= current_hour < 12:
        return "Good morning",current_day
    elif 12 <= current_hour < 17:
        return "Good afternoon",current_day
    elif 17 <= current_hour < 21:
        return "Good evening",current_day
    else:
        return "Hello",current_day

#uses regex to extract date,time and party size
def extract_entities(user_prompt):
    

    # Pattern matching for party size (e.g., "table for 2")
    party_size = None
    party_size_pattern = re.search(r"(table for|party of|for|party size to) (\d+)\b(?!\/-)", user_prompt, re.IGNORECASE)
    if party_size_pattern:
        party_size = int(party_size_pattern.group(2))


    time = check_time(user_prompt)

    date = None
    date_pattern = re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", user_prompt)

    if date_pattern:
        date = date_pattern.group(0)

    # Combine results
    extracted_info = {
        "party_size": party_size,
        "time": time,
        "date": date,
    }

    return extracted_info


def input_classifier():
    #preprocess X but keep the intents as it is
    X = [preprocess_text(prompt) for prompt in prompts]
    y = intents

        # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #vectorise the text 
    tfidf_vect = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, min_df=2, max_df=0.95)
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    X_test_tfidf = tfidf_vect.transform(X_test)

        #train the classifier
    classifier = LogisticRegression(max_iter=500, class_weight="balanced")
    classifier.fit(X_train_tfidf, y_train)

    evaluate_classifier(classifier, X_test_tfidf, y_test, "Intent Classifier")
    cross_validate_classifier(classifier, X_train_tfidf, y_train, "Intent Classifier") 
    

    return tfidf_vect, classifier
def sentiment_classifier():
    mood_sen = [preprocess_text(prompt) for prompt in mood_sentences]
    X_train, X_test, y_train, y_test = train_test_split(mood_sen, mood_intents, test_size=0.2, random_state=42)

    #TF-IDF vectorization
    tfidf_vect = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    X_test_tfidf = tfidf_vect.transform(X_test)

    # train the classifier
    classifier = LogisticRegression(max_iter=500, class_weight="balanced")
    classifier.fit(X_train_tfidf, y_train)
    
    evaluate_classifier(classifier, X_test_tfidf, y_test, "Sentiment Classifier")
    cross_validate_classifier(classifier, X_train_tfidf, y_train, "Sentiment Classifier")
    
    return tfidf_vect, classifier

# fucnction to get predicted intent for sentiment
def predict_mood(user_input):
    threshold=0.3
    tfidf_vect, classifier = sentiment_classifier()  
    input_tfidf = tfidf_vect.transform([user_input])
    
    #get prediction probabilities
    probabilities = classifier.predict_proba(input_tfidf)
    predicted_class_index = np.argmax(probabilities)
    confidence = probabilities[0][predicted_class_index]
    
    #compare to threshold
    if confidence >= threshold:
        mood = classifier.classes_[predicted_class_index]
    else:
        mood = "uncertain"
    return mood

#functions to get prediction for restaurent 
def predict_intent(user_prompt):
    threshold = 0.3
    tfidf_vect, classifier = input_classifier()
    user_prompt_tfidf = tfidf_vect.transform([user_prompt])
    predicted_type = classifier.predict(user_prompt_tfidf)[0]
    confidence = np.max(classifier.predict_proba(user_prompt_tfidf)) 
    
    if(confidence >= threshold):
        return predicted_type
    
    #fallback to cosine 
    if confidence < threshold:
        prompts_tfidf = tfidf_vect.transform(intents_df['prompt'])
        user_prompt_tfidf = tfidf_vect.transform([user_prompt])
        similarities = cosine_similarity(user_prompt_tfidf, prompts_tfidf).flatten()
        best_match_index = similarities.argmax()
        if similarities[best_match_index] > threshold:
            predicted_type = intents_df.iloc[best_match_index]['intent']
            return predicted_type
    
    

    return None

# extract user name using NER
def extract_name(sentence):
    sentence = " ".join(word.capitalize() for word in sentence.split())
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    for subtree in named_entities:
        if isinstance(subtree, Tree) and subtree.label() == 'PERSON':
            return " ".join(token for token, pos in subtree.leaves())
    return None

#function to check reservation conflicts
def check_reservation_conflict(date, time, party_size):
    for reservation in reservations_data["reservations"]:
        if reservation["date"] == date and reservation["time"] == time:
            if reservation["party_size"] + party_size > 10:  # Example maximum capacity
                return True
    return False

def format_reservation_details(reservation):
    return (
        f"Reservation ID: {reservation['reservation_id']}, "
        f"Name: {reservation['name']}, "
        f"Date: {reservation['date']}, "
        f"Time: {reservation['time']}, "
        f"Party Size: {reservation['party_size']}, "
    )


def get_reservation_by_id(reservation_id):
    for reservation in reservations_data["reservations"]:
        if reservation["reservation_id"] == reservation_id:
            return reservation
    return None  

#performance testing for intent and sentiment classifiers
def run_performance_tests():
    

    print("=== Intent Classifier ===\n")
    intent_vectorizer, intent_classifier = input_classifier()

    print("=== Sentiment Classifier ===\n")
    mood_vectorizer, mood_classifier = sentiment_classifier()

    print("Performance Tests Completed.\n")


# main function handles the transactions
def restaurant_bot():
    print("Chatbot: Hello! Welcome to the restaurant booking assistant. What's your name?")
    user_prompt = input("User: ")
    tmp_name = extract_name(user_prompt)
    greeting, current_day = personal_greeting()
    # Fallback if the name extraction fails
    if not tmp_name:
        tmp_name = "User"
    user_name = tmp_name
    print(f"Chatbot: {greeting}, {user_name}. Hope you are having a great {current_day}! How may I assist you today?")
    
    while True:
        user_prompt = input(f"{user_name}: ").strip()
        mood = predict_mood(user_prompt)

        matched_intent = predict_intent(user_prompt)
        
        #regex to see if users want to change name
        name_pattern = re.search(r"(?:call me|my name is|change my name to|you can call me)\s+(.+)", user_prompt, re.IGNORECASE)
        if name_pattern:
            user_name = name_pattern.group(1)
            print(f"Chatbot: I'll call you {user_name} from now on.")
            continue 
        
        #booking transaction        
        if matched_intent == "BookTable":
            extract_data = extract_entities(user_prompt)
            date = extract_data.get('date')
            time = extract_data.get('time')
            party_size = extract_data.get('party_size')
            booking_steps = 3
            #check if date is valid
            if(date):
                booking_steps -= 1
                for i in range(2):
                    date_bool, date_mes = check_date(date)
                    if(not date_bool):
                        print(date_mes)
                        date = input(f"{user_name}: ").strip()
                    else:
                        break
            #contextual tracker
            if(time):
                booking_steps -=1
            
            if(party_size):
                booking_steps -= 1
            
            #get date from user
            if(not date):
                print(f"Chatbot: Sure, {user_name}! Let's make a reservation. What date are you looking to book? (Format: DD-MM-YYYY)")
                date = input(f"{user_name}: ").strip()
                
                for i in range(2):
                    date_bool, date_mes = check_date(date)
                    if(not date_bool):
                        print(date_mes)
                        date = input(f"{user_name}: ").strip()
                    else:
                        if booking_steps > 1:
                            booking_steps -= 1
                            print(f"Chatbot: Only {booking_steps} more step, {user_name}")
                        break
                if(not date_bool):
                    print(f"Chatbot: Sorry {user_name} I'm afraid I don't understand please try rebooking")
                    continue            
                
            #get time from user
            if(not time):
                print(f"Chatbot: What time would you like to book? (Formats: HH:MM AM/PM or HH:MM in 24-hour format)")
                time = check_time(input(f"{user_name}: ").strip())
                
                for i in range(2):
                    if not time:
                        print(f"Chatbot: {user_name} please enter a vaild time (Formats: HH:MM AM/PM or HH:MM in 24-hour format)")
                        time = check_time(input(f"{user_name}: ").strip())
                    else:
                        if booking_steps > 1:
                            booking_steps -= 1
                            print(f"Chatbot: Only {booking_steps} more step, {user_name}")
                        break
                
                if not time:
                    print(f"Chatbot: Sorry {user_name} I'm afraid I don't understand please try rebooking")
                    continue

            #get party size from user
            if(not party_size or party_size <= 0):
                print(f"Chatbot: Please enter a valid number of people, {user_name} (min of 1).")
                for i in range(2):
                    try:
                        party_size = int(input(f"{user_name}: ").strip())
                        if party_size < 1:
                            raise ValueError("Invalid party size, it has to atleast 1")
                        if party_size > 10:
                            print(f"Chatbot: Sorry we can only hold a maximum of 10 party per booking")
                        else:
                            if booking_steps > 1:
                                booking_steps -= 1
                                print(f"Chatbot: Only {booking_steps} more step, {user_name}")
                            break
                    except ValueError:
                        print(f"Chatbot: Please enter a valid number of people, {user_name}.")
                        continue
            #party size cant be > 10
            if(party_size > 10):
                print(f"Chatbot: Sorry we can only hold a maximum of 10 party per booking")
                for i in range(2):
                    try:
                        party_size = int(input(f"{user_name}: ").strip())
                        if party_size < 1:
                            raise ValueError("Chatbot: Invalid party size, it has to atleast 1")
                        if party_size > 10:
                            print(f"Chatbot: Sorry we can only hold a maximum of 10 party per booking")
                        else:
                            break
                    except ValueError:
                        print(f"Chatbot: Please enter a valid number of people, {user_name}.")
                        continue
            #check for conflict with other reservations
            if check_reservation_conflict(date, time, party_size):
                print(f"Chatbot: Sorry, {user_name}, we don’t have enough capacity at that time. Would you like to try a different time?")
            else:
                exisiting_ids = [reservation["reservation_id"] for reservation in reservations_data["reservations"]]
                reservation_id = random.randint(1000, 9999)
                while reservation_id in exisiting_ids:
                    reservation_id = random.randint(1000, 9999)

                reservations_data["reservations"].append({
                    "reservation_id": reservation_id,
                    "name": user_name,
                    "date": date,
                    "time": time,
                    "party_size": party_size
                    
                })
                print(f"{user_name}! Here are your reservation details:")
                print(f"{format_reservation_details(get_reservation_by_id(reservation_id))}")
                print(f"Chatbot: Almost done now, {user_name}")
                print("Chatbot: Please confirm your booking by typing 'yes' to confirm or 'no' to continue")
                confirm_booking = input(f"{user_name}: ").strip().lower()
                if(confirm_booking == "yes"):
                    with open(reservation_json, 'w') as file:
                        json.dump(reservations_data, file, indent=4)
                    
                    print(f"Chatbot: {user_name}, your booking is confirmed, our team is looking forward to see you on {date}!")
                else:
                    print(f"Chatbot: No problem, {user_name}. Let me know if there's anything else I can assist you with!")
                    continue    
                    
                
                
        
        elif matched_intent == "CancelBooking":
            #confirm id and cancel
            print(f"Chatbot: Please provide your reservation ID, {user_name}.")
            try:
                res_id = int(input(f"{user_name}: ").strip())
                for reservation in reservations_data["reservations"]:
                    if reservation["reservation_id"] == res_id:
                        reservations_data["reservations"].remove(reservation)
                        print(f"Chatbot: Please type yes to confirm you'd like to cancel or no to continue")
                        confirm_cancel = input(f"{user_name}: ").strip()
                        if("yes" in confirm_cancel):
                            with open(reservation_json, 'w') as file:
                                json.dump(reservations_data, file, indent=4)
                            print(f"Chatbot: Your reservation has been successfully cancelled, {user_name}.")
                        else:
                            print(f"Chatbot: Glad to see your still with us {user_name}, looking forward to seeing you on {reservation['date']}!")
                            print("Chatbot: Anything else I can help with?")
                        break
                        
                            
                else:
                    print(f"Chatbot: Reservation ID not found, {user_name}. Please try again")
            except ValueError:
                print(f"Chatbot: Please enter a valid reservation ID, {user_name}.")
            

        elif matched_intent == "ModifyBooking":
            print(f"Chatbot: Please provide your reservation ID, {user_name}.")
            for i in range(2):
                try:
                    res_id = int(input(f"{user_name}: ").strip())
                    reservation = get_reservation_by_id(res_id)
                    if reservation:
                        print(f"Chatbot: Your reservation ID is confirmed, {user_name}. Here are the current details:")
                        print(format_reservation_details(reservation))
                        
                        print(f"Chatbot: Please tell me what details you'd like to modify. You can specify multiple details in one sentence (e.g., 'Change the date to 20-12-2024 and time to 7 PM').")
                        user_prompt = input(f"{user_name}: ").strip()
                        modifications = extract_entities(user_prompt)
                        new_date = reservation['date']
                        if modifications.get('date'):
                            # Handle date modification
                            new_date = modifications.get('date', reservation['date'])
                            valid_date, date_message = check_date(new_date)
                            if not valid_date:
                                print(f"Chatbot: {date_message}")
                                new_date = reservation['date']
                        
                        new_time = reservation['time']
                        if modifications.get('time'):
                            # Handle time modification
                            new_time = modifications.get('time', reservation['time'])
                            if not new_time:
                                new_time = reservation['time']

                        new_party_size = reservation['party_size']
                        if modifications.get('party_size'):
                            # Handle party size modification
                            new_party_size = modifications.get('party_size', reservation['party_size'])
                            if new_party_size <= 0:
                                print("Chatbot: Invalid party size, it has to atleast 1")
                                new_party_size = reservation['party_size']
                            if new_party_size > 10:
                                print(f"Chatbot: Sorry we can only hold a maximum of 10 party per booking")
                                new_party_size = reservation['party_size']
                                

                        # Check for conflicts
                        if check_reservation_conflict(new_date, new_time, new_party_size):
                            print(f"Chatbot: Sorry, {user_name}, there is a conflict with another reservation at the same time. Please choose a different time or date.")
                        else:
                            # Apply changes
                            reservation['date'] = new_date
                            reservation['time'] = new_time
                            reservation['party_size'] = new_party_size

                            print(f"{user_name}! Here are your modified reservation details:")
                            print(f"{format_reservation_details(get_reservation_by_id(res_id))}")
                            print("Chatbot: Please confirm your modification by typing 'yes' to confirm or 'no' to continue")
                            confirm_booking = input(f"{user_name}: ").strip().lower()
                            if(confirm_booking == "yes"):
                                with open(reservation_json, 'w') as file:
                                    json.dump(reservations_data, file, indent=4)
                                
                                print(f"Chatbot: {user_name}, your booking is updated, our team is looking forward to see you on {reservation['date']}!")
                                break
                            else:
                                print(f"Chatbot: No problem, {user_name}. Let me know if there's anything else I can assist you with!")
                                break    
                    else:
                        print(f"Chatbot: Reservation ID not found, {user_name}. Please try again.")
                
                except ValueError:
                    print(f"Chatbot: Please enter a valid reservation ID, {user_name}.")



        elif matched_intent == "AskLocationDetails":
            print(f"Chatbot: Our restaurant is located at 123 Foodie Lane")

        elif matched_intent == "AskMenuDetails":
            print(f"Chatbot: We offer a variety of dishes, including vegan and gluten-free options")


        elif matched_intent == "ExitChat":
            #confirmation
            print(f"Chatbot: Are you sure you want to exit? Type 'yes' to confirm or 'no' to continue.")
            confirm_exit = input(f"{user_name}: ").strip().lower()
            if confirm_exit == 'yes':
                print(f"Chatbot: Thank you for using the restaurant booking assistant, {user_name}. Goodbye!")
                break
            else:
                print(f"Chatbot: Glad to have you back, {user_name}. How can I help you?")
        #help options
        elif matched_intent == "Help":
            print(f"Chatbot: Here are some things I can assist you with:\n"
                  "* Book a table: 'I want to book a table for 4 on 25-02-2025 at 7 PM or I want to book.'\n"
                  "* Cancel a reservation: 'Cancel my reservation.'\n"
                  "* Modify a reservation: 'Change my reservation or Modify my reservation.'\n"
                  "* Ask for location: 'Where is the restaurant located?'\n"
                  "* Ask about the menu: 'What's on the menu?'\n"
                  "* Exit the chat: 'I want to leave' or 'I want to exit'\n"
                  "* Small talk: Ask general questions or chat casually.\n"
                  "* Change name: 'change my name to smith  or call me smith\n"
                  "Note: These commands don't need to be worded exactly as shown; they are just examples of some of the many phrases that work.")
            continue
        #check small talk and Q&A 
        elif not matched_intent:
            response = search_document(small_talk_corpus,small_talk_answers,small_talk_matrix, small_talk_vectoriser, user_prompt)
            
            #output users name
            if response == "smalltalk_user_name":
                print(f"Chatbot: Your name is {user_name}")
            #directly output the answer    
            elif response and response != "smalltalk_user_name":
                print(f"Chatbot: {response}")
            else:
                #mood checks
                if mood == "angry":
                    print("Chatbot: I sense frustration ☹️. Here are some things I can help you with:")
                    print("* Book a table\n* Cancel a reservation\n* Modify a reservation\n* Ask for location\n* View the menu")
                    print("Please try asking again, or ask for help to see all options.")
                elif mood == "happy":
                    print(f"Chatbot: Glad you're feeling good {user_name} 😊, Here are some things I can help you with:")
                    print("* Book a table\n* Cancel a reservation\n* Modify a reservation\n* Ask for location\n* View the menu")
                    print("Please try asking again, or ask for help to see all options.")
                elif mood == "neutral":
                    print("Chatbot: Here are some things I can help you with:")
                    print("* Book a table\n* Cancel a reservation\n* Modify a reservation\n* Ask for location\n* View the menu")
                    print("Please try asking again, or ask for help to see all options.")
                elif mood == "uncertain":    
                    print(f"Chatbot: I'm not sure how to respond to that {user_name}. Here are some things I can help you with:")
                    print("* Book a table\n* Cancel a reservation\n* Modify a reservation\n* Ask for location\n* View the menu")
                    print("Please try asking again, or ask for help to see all options.")

# Main Execution
if __name__ == "__main__":
    run_performance_tests()
    restaurant_bot()
    
