import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import nltk
import asyncio
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('wordnet', quiet=True)

# Set stopwords
STOPWORDS = set(stopwords.words('english'))

# Set page configuration
st.set_page_config(layout="wide")

# Load CSS styles
def load_css():
    st.markdown("""
        <style>
            body {
                color: #D2C9A3;               
                font-family: Arial, sans-serif;
            }
            
            h1, h2, h3, h4, h5, h6 {
                color: brown;               
                text-transform: uppercase; 
            }
            .stSidebar {
                background-color: black;  
                color: #D2C9A3;   
                text-align: center;
                font-size: 10px;
            }
            .stButton > button {
                background-color: #D2C9A3;   
                color: black;             
                border: none;             
                padding: 10px;           
                border-radius: 5px;     
                cursor: pointer;         
                width: 100%;             
                text-align: center;      
                font-size: 16px;         
            }
            .stButton > button:hover {
                background-color: brown;
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)

# Load CSS
load_css()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'page' not in st.session_state:
    st.session_state.page = "Home"

def initialize_database(db_name, create_statement):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(create_statement)
        conn.commit()  # Commit the changes
    except Exception as e:
        st.error(f"Error initializing database {db_name}: {e}")
    finally:
        conn.close()

# Create tables with proper SQL syntax
initialize_database(
    'uploaded_sentiment_analysis.db',
    '''
    CREATE TABLE IF NOT EXISTS uploaded_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT,
        user_id TEXT,
        profile_name TEXT,
        helpfulness_numerator INTEGER,
        helpfulness_denominator INTEGER,
        score INTEGER,
        time INTEGER,
        summary TEXT,
        text TEXT,
        processed_text TEXT,
        sentiment TEXT
    )
    '''
)

initialize_database(
    'scraped_sentiment_analysis.db',
    '''
    CREATE TABLE IF NOT EXISTS scraped_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        author TEXT,
        rating TEXT,
        title TEXT,
        content TEXT,
        date TEXT,
        verified TEXT
    )
    '''
)  

# Set up sentiment analysis
sia = SentimentIntensityAnalyzer()

def preprocess_text(text):
    return text.lower().strip().replace('\n', ' ')

def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score > 0.05:
        return 'Positive'
    elif compound_score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def filter_unwanted_comments(reviews, unwanted_keywords):
    filtered_reviews = []
    for review in reviews:
        if not any(keyword in review['content'].lower() for keyword in unwanted_keywords):
            filtered_reviews.append(review)
    return filtered_reviews

def insert_uploaded_output_review(product_id, user_id, profile_name, helpfulness_numerator, helpfulness_denominator, score, time, summary, text, processed_text, sentiment):
    try:
        conn = sqlite3.connect('uploaded_sentiment_analysis.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO uploaded_reviews (product_id, user_id, profile_name, helpfulness_numerator, helpfulness_denominator, score, time, summary, text, processed_text, sentiment) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (product_id, user_id, profile_name, helpfulness_numerator, helpfulness_denominator, score, time, summary, text, processed_text, sentiment))
        conn.commit()
    except Exception as e:
        st.error(f"Error inserting review into uploaded_reviews: {e}")
    finally:
        conn.close()

def chat_and_help_section():
    st.title("Chat & Help Assistant")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I assist you? (Type 'help' for options)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = generate_response(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

def generate_response(prompt):
    prompt = prompt.lower()  
    help_responses = {
        'scrape reviews': (
            "### Scrape Reviews\n"
            "To scrape reviews from Amazon, please follow these steps:\n"
            "1. Enter the Amazon product ASIN in the input field.\n"
            "2. Click the 'SCRAPE REVIEWS' button.\n"
            "3. Review the scraped data displayed below.\n"
            "Ensure the ASIN is correct for accurate results!"
        ),
        'dataset management': (
            "### Dataset Management\n"
            "To upload a CSV dataset for analysis:\n"
            "1. Make sure your file contains the necessary columns (Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text).\n"
            "2. Click 'UPLOAD DATASET' to analyze sentiment and detect fake reviews."
        ),
        'text analysis': (
            "### Text Analysis\n"
            "You can analyze the sentiment of any text by entering it in the provided text area.\n"
            "The system will categorize the sentiment as Positive, Negative, or Neutral and provide sentiment scores."
        ),
        'fake review detection': (
            "### Fake Review Detection\n"
            "1. Upload a CSV file containing product reviews.\n"
            "2. The system will analyze the reviews to determine if they are likely fake or genuine based on sentiment and rating."
        ),
        'history': (
            "### Review History\n"
            "You can view all previously scraped and uploaded datasets here.\n"
            "Filter, download, and analyze the sentiment of past reviews for more insights."
        ),
        'help': (
            "### Need Assistance?\n"
            "Feel free to ask me about any of these features:\n"
            "- Scrape reviews from Amazon\n"
            "- Upload a dataset for sentiment analysis\n"
            "- Conduct text analysis for custom inputs\n"
            "Type 'Getting Started' for a quick overview of what to do next!"
        ),
        'getting started': (
            "### Getting Started\n"
            "Here's how to get started with SentimelyzeR:\n"
            "1. Scrape product reviews by going to 'Scrape Reviews'.\n"
            "2. Analyze your own datasets by uploading them in 'Dataset Management'.\n"
            "3. Use 'Text Analysis' for direct sentiment evaluation of any text you provide."
        ),
        'tutorial': (
            "### Interactive Tutorial\n"
            "Explore the following features of this dashboard:\n"
            "1. **Scrape Reviews**: Gather reviews from Amazon using the product ASIN.\n"
            "2. **Dataset Management**: Upload and analyze your own review datasets.\n"
            "3. **Text Analysis**: Get sentiment analysis results for any text inscription.\n"
            "Use the sidebar to navigate between different sections for a comprehensive analysis!"
        )
    }

    response = help_responses.get(prompt, 
    "I couldn’t understand your question. You can type:\n- Scrape Reviews\n- Upload Dataset\n- Text Analysis\n- Getting Started\n\n"
    "Or type 'help' for options and further assistance.")

    return response

def get_soup(url):
    custom_headers = {
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,/;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "Gecko/20100101 Firefox/135.0"
        )
    }
    try:
        response = requests.get(url, headers=custom_headers)
        response.raise_for_status()  
        return BeautifulSoup(response.text, "lxml")
    except requests.RequestException as e:
        st.error(f"Error fetching the webpage: {e}")
        return None

def extract_review(review):
    author = review.select_one(".a-profile-name").text.strip() if review.select_one(".a-profile-name") else "Unknown"
    rating = review.select_one(".review-rating > span").text.replace("out of 5 stars", "").strip() if review.select_one(".review-rating > span") else "No Rating"
    date = review.select_one(".review-date").text.strip() if review.select_one(".review-date") else "No Date"
    title = review.select_one(".review-title span:not([class])").text.strip() if review.select_one(".review-title span:not([class])") else "No Title"
    content = ' '.join(review.select_one(".review-text").stripped_strings) if review.select_one(".review-text") else "No Content"
    verified_element = review.select_one("span.a-size-mini")
    verified = verified_element.text.strip() if verified_element else "Not Verified"

    return {
        "author": author,
        "rating": rating,
        "title": title,
        "content": content.replace("Read more", ""),
        "date": date,
        "verified": verified
    }

def get_reviews(soup):
    reviews = []
    local_reviews = soup.select("#cm-cr-dp-review-list > li")
    global_reviews = soup.select("#cm-cr-global-review-list > li")

    for review in local_reviews:
        reviews.append(extract_review(review))
        
    for review in global_reviews:
        reviews.append(extract_review(review))

    return reviews

async def scrape_reviews(asin):
    reviews = []
    search_url = f"https://www.amazon.com/dp/{asin}"
    soup = get_soup(search_url)
    if soup:
        reviews = get_reviews(soup)

    unwanted_keywords = ['fake', 'unverified', 'not helpful', 'spam']
    return filter_unwanted_comments(reviews, unwanted_keywords)

def generate_insights(data):
    positive_reviews = data[data['Sentiment'] == 'Positive']
    negative_reviews = data[data['Sentiment'] == 'Negative']
    neutral_reviews = data[data['Sentiment'] == 'Neutral']

    insights = [
        f"{len(positive_reviews)} positive reviews found.",
        f"{len(negative_reviews)} negative reviews found.",
        f"{len(neutral_reviews)} neutral reviews found."
    ]
    return insights

def insert_scraped_review(author, rating, title, content, date, verified):
    try:
        conn = sqlite3.connect('scraped_sentiment_analysis.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scraped_reviews (author, rating, title, content, date, verified) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (author, rating, title, content, date, verified))
        conn.commit()
    except Exception as e:
        st.error(f"Error inserting review into scraped_reviews: {e}")
    finally:
        conn.close()

def fetch_all_reviews(table_name):
    db_name = 'scraped_sentiment_analysis.db' if 'scraped' in table_name else 'uploaded_sentiment_analysis.db'
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM {table_name}')
        data = cursor.fetchall()
    except Exception as e:
        st.error(f"Error fetching reviews from {table_name}: {e}")
        return []
    finally:
        conn.close()
    return data

def clear_database(table_name):
    db_name = 'scraped_sentiment_analysis.db' if 'scraped' in table_name else 'uploaded_sentiment_analysis.db'

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(f'DELETE FROM {table_name}') 
        conn.commit()
        st.success(f"All entries in {table_name} have been cleared.")
    except Exception as e:
        st.error(f"Error clearing {table_name}: {e}")
    finally:
        conn.close()

def export_to_csv(data, filename):
    csv = data.to_csv(index=False)
    st.download_button(label="Download Results as CSV", data=csv, file_name=filename, mime='text/csv')

def show_tutorial():
    st.subheader("Interactive Tutorial")
    st.markdown("""
    Welcome to the Sentiment Analysis Dashboard! Here’s how to get started:
    - **Home**: Overview of functionalities and quick insights.
    - **Dataset Management**: Analyze your own CSV files for sentiment and identify fake reviews.
    - **Text Analysis**: Input custom text to understand sentiment.
    - **Scrape Reviews**: Collect Amazon product reviews with ease.
    - **History**: View past analyses and results for comparison.

    **Explore More** using the sidebar to navigate through the features of this dashboard!
    """)
    st.markdown("### Quick Tips:")
    st.markdown("- Use the clear buttons in History to manage your databases.")
    st.markdown("- Visualizations provide clear insights into data trends and sentiments.")

def clean_review(review):
    stp_words = stopwords.words('english')
    cleanreview = " ".join(word for word in review.split() if word not in stp_words)
    return cleanreview 

def train_svm_model(data):
    data.dropna(inplace=True)
    data['Sentiment'] = pd.to_numeric(data['Sentiment'], errors='coerce') 
    data.dropna(subset=['Sentiment'], inplace=True)  

    data.loc[data['Sentiment'] <= 3, 'Sentiment'] = 0
    data.loc[data['Sentiment'] > 3, 'Sentiment'] = 1 

    data['Review'] = data['Review'].astype(str).apply(clean_review)  
    data = data[data['Review'].str.strip().astype(bool)]

    cv = TfidfVectorizer(max_features=2500)
    if not data['Review'].empty:
        X = cv.fit_transform(data['Review']).toarray()
        x_train, x_test, y_train, y_test = train_test_split(X, data['Sentiment'], test_size=0.25, random_state=42)

        model = SVC(kernel='linear')
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, pred)
        st.write(f'Model Accuracy: {accuracy:.4f}')

        cm = confusion_matrix(y_test, pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        
        plt.figure(figsize=(8, 6))
        cm_display.plot()
        st.pyplot(plt)
    else:
        st.error("No valid reviews available for training the model.")

def text_analysis_section():
    st.title("Text Analysis")
    user_input = st.text_area("Enter the text to analyze:", height=200)

    translator = Translator()

    if st.button("Analyze Sentiment"):
        if user_input:
            # Detect language
            detected_language = translator.detect(user_input).lang
            
            # Translate to English if not already
            if detected_language != 'en':
                translated_text = translator.translate(user_input, dest='en').text
                st.write(f"**Translated Text:** {translated_text}")
            else:
                translated_text = user_input
            
            # Process and analyze sentiment
            processed_text = preprocess_text(translated_text)
            sentiment = analyze_sentiment(processed_text)
            st.success(f"The sentiment analysis result is: **{sentiment}**")
            
            # Display sentiment scores
            scores = sia.polarity_scores(processed_text)
            st.write("### Sentiment Scores:")
            st.json(scores)  
        else:
            st.error("Please enter text for analysis!")

def display_navbar():
    st.sidebar.write("") 
    st.sidebar.image('sentimelyzer.png', use_container_width=True)
    st.sidebar.write("") 
    st.sidebar.write("") 
    if st.sidebar.button("Home", key="home_button"):
        st.session_state.page = "Home"
    if st.sidebar.button("About", key="about_button"):
        st.session_state.page = "About"
    if st.sidebar.button("Scrape Reviews", key="scrape_reviews_button"):
        st.session_state.page = "Scrape Reviews"
    if st.sidebar.button("Dataset Management", key="dataset_management_button"):
        st.session_state.page = "Dataset Management"
    if st.sidebar.button("Text Analysis", key="text_analysis_button"):
        st.session_state.page = "Text Analysis"
    if st.sidebar.button("History", key="history_button"):
        st.session_state.page = "History"
    if st.sidebar.button("Support", key="support_button"):
        st.session_state.page = "Support"

display_navbar()

if st.session_state.page == "Home":
    st.markdown("""
                <h1></h1><h1></h1>
        <h1 style="text-align: center; font-size: 80px;">Welcome To SentimelyzeR</h1>
        <h3 style="text-align: center; font-size: 24px;">Your Gateway to Understanding Sentiments</h3>
    """, unsafe_allow_html=True)

if st.session_state.page == "About":
    st.title("About")
    st.markdown("""
        Sentimelyzer is your go-to tool for effortlessly analyzing Amazon product reviews to uncover customer sentiment. 
        With features like scraping reviews directly from Amazon URLs, uploading your own CSV datasets for analysis, and 
        evaluating custom text inputs, you can gain valuable insights into consumer opinions. 
        Our interactive visualizations make it easy to interpret data trends and improve your product offerings based on real feedback.

        ### Key Features:
        - **Scrape Reviews**: Instantly gather reviews from Amazon products using the ASIN.
        - **Dataset Management**: Upload your own CSV files to analyze sentiments and detect fake reviews.
        - **Text Analysis**: Analyze the sentiment of any custom text input.
        - **Data Visualization**: Gain insights through comprehensive pie and bar charts depicting sentiment distribution.
    """)

if st.session_state.page == "Support":
    st.subheader("Support")
    st.markdown("""
        ### Need assistance?
        Our assistance bot is here to help you navigate through:
        - How to scrape reviews from Amazon using ASIN.
        - Instructions to upload datasets for sentiment analysis and fake review detection.
        - Guidance on conducting text sentiment analysis.
        - Accessing your analysis history and managing datasets effectively.

        If you have a specific query, simply type in your question below, or explore our available options by typing 'help'.
    """)
    chat_and_help_section()

if st.session_state.page == "Scrape Reviews":
    st.header("SCRAPE REVIEWS FROM AMAZON")
    asin_input = st.text_input("ENTER AMAZON ASIN:")

    if st.button("SCRAPE REVIEWS", key="start_scrape"):
        if asin_input:
            with st.spinner('SCRAPING DATA...'):
                scraped_reviews = asyncio.run(scrape_reviews(asin_input))
            st.success("DATA SCRAPING COMPLETE!")

            df_reviews = pd.DataFrame(scraped_reviews)
            st.write("### SCRAPED REVIEWS")
            st.write(df_reviews)

            if not df_reviews.empty:
                for _, row in df_reviews.iterrows():
                    insert_scraped_review(row['author'], row['rating'], row['title'], 
                                          row['content'], row['date'], 
                                          row['verified'])

                export_to_csv(df_reviews, "scraped_reviews.csv")

                st.write("### SENTIMENT DISTRIBUTION")
                df_reviews['Sentiment'] = df_reviews['content'].apply(analyze_sentiment)
                sentiment_counts = df_reviews['Sentiment'].value_counts()
                sentiment_counts_df = pd.DataFrame(sentiment_counts).reset_index()
                sentiment_counts_df.columns = ['Sentiment', 'Counts']

                st.write("#### Pie Chart of Sentiment Distribution")
                fig_pie = go.Figure(data=[go.Pie(labels=sentiment_counts_df['Sentiment'],
                                                   values=sentiment_counts_df['Counts'],
                                                   hole=0.3,
                                                   marker=dict(colors=['green' if sentiment == 'Positive'
                                                                       else 'red' if sentiment == 'Negative'
                                                                       else 'yellow'
                                                                       for sentiment in sentiment_counts_df['Sentiment']]))])
                st.plotly_chart(fig_pie)

                positive_reviews_text = ' '.join(df_reviews[df_reviews['Sentiment'] == 'Positive']['content'])
                negative_reviews_text = ' '.join(df_reviews[df_reviews['Sentiment'] == 'Negative']['content'])

                with st.container():
                    st.write("### WORD CLOUD FOR REVIEWS")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Positive Reviews Word Cloud")
                        if positive_reviews_text:
                            plt.figure(figsize=(4, 4))
                            wordcloud_pos = WordCloud(width=200, height=200, background_color='black').generate(positive_reviews_text)
                            plt.imshow(wordcloud_pos, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(plt)
                            plt.close()
                        else:
                            st.write("*No positive reviews available to generate a word cloud.*")

                    with col2:
                        st.subheader("Negative Reviews Word Cloud")
                        if negative_reviews_text:
                            plt.figure(figsize=(4, 4))
                            wordcloud_neg = WordCloud(width=200, height=200, background_color='black').generate(negative_reviews_text)
                            plt.imshow(wordcloud_neg, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(plt)
                            plt.close()
                        else:
                            st.write("*No negative reviews available to generate a word cloud.*")

                st.write("### Bar Chart of Ratings by Sentiment")
                rating_count = df_reviews.groupby(['Sentiment', 'rating']).size().reset_index(name='Counts')
                fig_bar = px.bar(rating_count, x='rating', y='Counts', color='Sentiment', barmode='group',
                                 title='Bar Chart of Ratings by Sentiment', 
                                 color_discrete_map={
                                     'Positive': 'green',
                                     'Negative': 'red',
                                     'Neutral': 'yellow'
                                 },
                                 labels={'Counts': 'Number of Reviews', 'Rating': 'Rating'})
                st.plotly_chart(fig_bar)

                insights = generate_insights(df_reviews)
                st.write("### INSIGHTS")
                for insight in insights:
                    st.write(insight)

            else:
                st.write("*NO REVIEWS FOUND DURING SCRAPING.*")

if st.session_state.page == "Dataset Management":
    st.header("DATASET MANAGEMENT")
    st.subheader("Upload Reviews for Sentiment Analysis and Fake Review Detection")

    uploaded_file = st.file_uploader("CHOOSE A CSV FILE WITH REVIEWS", type="csv")

    if uploaded_file is not None:
        fake_reviews_data = pd.read_csv(uploaded_file)

        st.write("### UPLOADED REVIEWS")
        st.write(fake_reviews_data)

        # Required columns check
        required_columns = ['Id', 'ProductId', 'UserId', 'ProfileName', 
                            'HelpfulnessNumerator', 'HelpfulnessDenominator', 
                            'Score', 'Time', 'Summary', 'Text']

        if all(col in fake_reviews_data.columns for col in required_columns):
            # Sentiment Analysis
            fake_reviews_data['Processed_Text'] = fake_reviews_data['Text'].apply(preprocess_text)
            fake_reviews_data['Sentiment'] = fake_reviews_data['Processed_Text'].apply(analyze_sentiment)

            for _, row in fake_reviews_data.iterrows():
                insert_uploaded_output_review(row['ProductId'], row['UserId'], row['ProfileName'], 
                                              row['HelpfulnessNumerator'], row['HelpfulnessDenominator'],
                                              row['Score'], row['Time'], row['Summary'], 
                                              row['Text'], row['Processed_Text'], row['Sentiment'])

            st.success("DATA UPLOADED AND SENTIMENT ANALYZED!")

            export_to_csv(fake_reviews_data, "uploaded_reviews.csv")

            st.write("### SENTIMENT DISTRIBUTION")
            sentiment_counts = fake_reviews_data['Sentiment'].value_counts()
            sentiment_counts_df = pd.DataFrame(sentiment_counts).reset_index()
            sentiment_counts_df.columns = ['Sentiment', 'Counts']

            st.write("#### Pie Chart of Sentiment Distribution")
            fig_pie = go.Figure(data=[go.Pie(labels=sentiment_counts_df['Sentiment'],
                                               values=sentiment_counts_df['Counts'],
                                               hole=0.3, 
                                               marker=dict(colors=['green' if sentiment == 'Positive' 
                                                                   else 'red' if sentiment == 'Negative'
                                                                   else 'yellow'
                                                                   for sentiment in sentiment_counts_df['Sentiment']]))])
            st.plotly_chart(fig_pie)

            # Fake Review Detection
            st.subheader("Fake Review Detection Results")

            # Function to analyze sentiment and classify as Fake/Real
            def classify_fake_reviews(row):
                sentiment_score = sia.polarity_scores(row['Text'])['compound']  
                rating = float(row['Score'])
                
                if rating < 3 and sentiment_score >= 0.05:
                    return 'Fake'
                elif rating >= 3:
                    return 'Real'
                return 'Uncertain'

            fake_reviews_data['Is_Fake'] = fake_reviews_data.apply(classify_fake_reviews, axis=1)
            st.write("#### Fake Review Detection Results")
            results_df = fake_reviews_data[['Id', 'ProductId', 'UserId', 'ProfileName', 'Score', 'Text', 'Is_Fake']].copy()
            st.write(results_df)

            fake_count = fake_reviews_data[fake_reviews_data['Is_Fake'] == 'Fake'].shape[0]
            real_count = fake_reviews_data[fake_reviews_data['Is_Fake'] == 'Real'].shape[0]
            uncertain_count = fake_reviews_data[fake_reviews_data['Is_Fake'] == 'Uncertain'].shape[0]

            st.write("### ANALYSIS INSIGHTS")
            st.write(f"Total Reviews: {len(fake_reviews_data)}")
            st.write(f"Fake Reviews Detected: {fake_count}")
            st.write(f"Real Reviews Detected: {real_count}")
            st.write(f"Uncertain Reviews: {uncertain_count}")

            fig = px.pie(names=['Fake', 'Real', 'Uncertain'],
                         values=[fake_count, real_count, uncertain_count],
                         title='Distribution of Fake vs Real Reviews',
                         color_discrete_sequence=['green', 'red', 'yellow'])
            st.plotly_chart(fig)

        else:
            st.write("**Uploaded CSV must contain the following columns:** Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text.")

if st.session_state.page == "History":
    st.header("History of Reviews")

    # Display scraped reviews
    st.subheader("Scraped Reviews")
    scraped_reviews = fetch_all_reviews('scraped_reviews')
    
    if scraped_reviews:
        df_scraped = pd.DataFrame(scraped_reviews, columns=["ID", "Author", "Rating", "Title", "Content", "Date", "Verified"])
        st.write("### ALL SCRAPED REVIEWS")
        st.write(df_scraped)
    else:
        st.write("*NO SCRAPED REVIEWS FOUND.*")
    if st.button("Clear Scraped Reviews Database"):
        clear_database('scraped_reviews')  

    # Display uploaded output reviews
    st.subheader("Uploaded Output Reviews")
    uploaded_output_reviews = fetch_all_reviews('uploaded_reviews')
    
    if uploaded_output_reviews:
        df_uploaded_output = pd.DataFrame(uploaded_output_reviews, columns=["ID", "ProductId", "UserId", "ProfileName", 
                                                                           "HelpfulnessNumerator", "HelpfulnessDenominator", 
                                                                           "Score", "Time", "Summary", "Text", "Processed_Text", "Sentiment"])
        st.write("### ALL UPLOADED OUTPUT REVIEWS")
        st.write(df_uploaded_output)
    else:
        st.write("*NO UPLOADED OUTPUT REVIEWS FOUND.*")

    if st.button("Clear Uploaded Output Reviews Database"):
        clear_database('uploaded_reviews')  

# Text Analysis Section
if st.session_state.page == "Text Analysis":
    text_analysis_section()
