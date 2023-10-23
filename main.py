import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Example comments
# Read comments from an Excel file
'''comments_df = pd.read_excel("comments.xlsx")
comments = comments_df.iloc[:, 0].tolist()'''


comments = [
    "The accessibility and availability are excellent.",
    "The amenities and location are top-notch.",
    "The speed and efficiency are impressive.",
    "The reliability and maintenance are satisfactory.",
    "The ease of use is fantastic.",
    "The compatibility and connectivity need improvement.",
    "The customer service is outstanding.",
    "The price and cost are reasonable.",
    "The user-interface and mobile app are user-friendly.",
    "The queue and waiting time are too long.",
    "The safety measures are commendable.",
    "The payment options are limited.",
]

# Define aspects of interest
aspects = [
    "Accessibility+Availability",
    "Amenities+Location",
    "Speed+Efficiency",
    "Reliability+Maintenance",
    "Ease of use",
    "Compatibility+Connectivity",
    "Customer Service",
    "Price+Cost",
    "User-Interface+Mobile App",
    "Queue+Waiting Time",
    "Safety",
    "Payment Options",
]

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Preprocess comments
stop_words = set(stopwords.words("english"))

def preprocess(comment):
    tokens = word_tokenize(comment.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(tokens)

preprocessed_comments = [preprocess(comment) for comment in comments]

# Perform sentiment analysis using few-shot approach
results = {}

# Few-shot labeled data for each aspect
labeled_data = {
    "Accessibility+Availability": [
        ("The accessibility is great.", "positive"),
        ("Availability is a major issue.", "negative"),
        ("The service is easily accessible.", "positive"),
    ],
    "Amenities+Location": [
        ("The amenities provided are excellent.", "positive"),
        ("Location is convenient for me.", "positive"),
        ("The amenities are lacking.", "negative"),
    ],
    "Speed+Efficiency": [
        ("The speed of the service is impressive.", "positive"),
        ("Efficiency needs improvement.", "negative"),
        ("The service is fast and efficient.", "positive"),
    ],
    "Reliability+Maintenance": [
        ("The service is reliable and well-maintained.", "positive"),
        ("The service suffers from frequent breakdowns.", "negative"),
        ("Reliability is a concern.", "negative"),
    ],
    "Ease of use": [
        ("The service is incredibly easy to use.", "positive"),
        ("I find it difficult to navigate.", "negative"),
        ("The service is user-friendly.", "positive"),
    ],
    "Compatibility+Connectivity": [
        ("The service is compatible with most devices.", "positive"),
        ("Connectivity issues need to be addressed.", "negative"),
        ("Compatibility is a problem.", "negative"),
    ],
    "Customer Service": [
        ("The customer service is excellent.", "positive"),
        ("I had a terrible experience with customer support.", "negative"),
        ("Customer service was prompt and helpful.", "positive"),
    ],
    "Price+Cost": [
        ("The price is reasonable for the service.", "positive"),
        ("The cost is too high.", "negative"),
        ("The service offers good value for the price.", "positive"),
    ],
    "User-Interface+Mobile App": [
        ("The user-interface is intuitive and user-friendly.", "positive"),
        ("I struggle with the mobile app.", "negative"),
        ("Navigating through the UI is easy.", "positive"),
    ],
    "Queue+Waiting Time": [
        ("The waiting time is too long.", "negative"),
        ("I didn't have to wait for long.", "positive"),
        ("The queue management needs improvement.", "negative"),
    ],
    "Safety": [
        ("The service takes safety seriously.", "positive"),
        ("Safety measures are lacking.", "negative"),
        ("I feel safe using the service.", "positive"),
    ],
    "Payment Options": [
        ("There are multiple payment options available.", "positive"),
        ("Limited payment options are inconvenient.", "negative"),
        ("I had issues with the payment process.", "negative"),
    ],
}

# Iterate over aspects
for aspect in aspects:
    aspect_sentiments = []
    
    # Few-shot training data for the aspect
    few_shot_data = labeled_data[aspect]
    
    # Train the sentiment analyzer on few-shot data
    for data_point in few_shot_data:
        comment, label = data_point
        preprocessed_comment = preprocess(comment)
        sentiment_scores = sid.polarity_scores(preprocessed_comment)
        
        if label == "positive":
            aspect_sentiments.append(sentiment_scores["compound"])
        elif label == "negative":
            aspect_sentiments.append(-sentiment_scores["compound"])
    
    # Perform sentiment analysis on target comments
    for comment in preprocessed_comments:
        sentiment_scores = sid.polarity_scores(comment)
        aspect_sentiments.append(sentiment_scores["compound"])
    
    avg_sentiment = sum(aspect_sentiments) / len(aspect_sentiments) if aspect_sentiments else 0.0
    results[aspect] = avg_sentiment

# Print results
for aspect, sentiment in results.items():
    print(f"Aspect: {aspect} | Sentiment: {sentiment}")