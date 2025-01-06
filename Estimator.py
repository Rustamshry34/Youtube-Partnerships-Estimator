import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QLabel, QLineEdit, QPushButton, QTableWidget,
                           QTableWidgetItem, QMessageBox, QProgressDialog,
                           QFrame, QHBoxLayout, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
import os 
import regex
import re
import isodate
import numpy as np


class EvaluationWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, channel_url, video_count):
        super().__init__()
        self.channel_url = channel_url
        self.video_count = video_count 
        self.api_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXX'
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.setup_nltk()
        

    def setup_nltk(self):
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                truncation=True,
                max_length=512
            )
        except Exception as e:
            print(f"Error loading multilingual model: {str(e)}")
            self.error.emit(f"Error loading multilingual model: {str(e)}")

    def run(self):
        try:
            channel_id = self.extract_channel_id(self.channel_url)
            channel_name = self.get_channel_name(channel_id)

            self.progress.emit("Evaluating channel...")
            overall_sponsored_score, overall_unsponsored_score, num_sponsored, num_unsponsored, sponsored_sentiment_counts, unsponsored_sentiment_counts, avg_sponsored_engagement_score, avg_unsponsored_engagement_score, avg_sponsored_sentiment, avg_unsponsored_sentiment = self.evaluate_channel(channel_id, self.video_count)
            
            results = {
                'channel_name': channel_name,
                'channel_id': channel_id,
                'sponsored_score': overall_sponsored_score,
                'unsponsored_score': overall_unsponsored_score,
                'num_sponsored': num_sponsored,
                'num_unsponsored': num_unsponsored,
                'Sponsored_Sentiment_Score': avg_sponsored_sentiment,
                'Unsponsored_Sentiment_Score': avg_unsponsored_sentiment,
                'Sponsored_Engagement_Score': avg_sponsored_engagement_score,
                'Unsponsored_Engagement_Score': avg_unsponsored_engagement_score, 
                'sponsored_positive_comments': sponsored_sentiment_counts['Positive'],
                'sponsored_negative_comments': sponsored_sentiment_counts['Negative'],
                'sponsored_neutral_comments': sponsored_sentiment_counts['Neutral'],
                'unsponsored_positive_comments': unsponsored_sentiment_counts['Positive'],
                'unsponsored_negative_comments': unsponsored_sentiment_counts['Negative'],
                'unsponsored_neutral_comments': unsponsored_sentiment_counts['Neutral']
            }
            
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


    def get_channel_videos(self, channel_id, video_count):
        self.progress.emit("Fetching channel videos...")
        videos = []
        next_page_token = None
        base_video_url = "https://www.youtube.com/watch?v="

        while len(videos) < video_count:
            request = self.youtube.search().list(
                part='snippet',
                channelId=channel_id,
                maxResults=min(2, video_count - len(videos)),
                type='video',
                order='date',
                pageToken=next_page_token
            ).execute()

            video_ids = [item['id']['videoId'] for item in request.get('items', []) if 'videoId' in item['id']]
            if not video_ids:
                break

            video_details_request = self.youtube.videos().list(
                part="contentDetails",
                id=",".join(video_ids)
            ).execute()

            for item, video_details in zip(request['items'], video_details_request['items']):
                if 'videoId' in item['id']:
                    video_id = item['id']['videoId']
                    video_title = item['snippet']['title']
                    video_url = f"{base_video_url}{video_id}"

                    duration = video_details['contentDetails']['duration']
                    parsed_duration = isodate.parse_duration(duration).total_seconds()

                    if parsed_duration < 180:
                        continue

                    videos.append({'video_id': video_id, 'video_title': video_title, 'video_url': video_url})

            next_page_token = request.get('nextPageToken')
            if not next_page_token:
                break

        return videos

    def get_video_details(self, video_id):
        request = self.youtube.videos().list(part='statistics', id=video_id).execute()
        details = request['items'][0]['statistics']
        return details

    def check_sponsorship_disclaimer(self, video_url):
        self.progress.emit(f"Checking sponsorship disclaimer for {video_url}")
        C_options = Options()
        C_options.add_argument('--headless') 
        C_options.add_argument('--no-sandbox')
        C_options.add_argument('--disable-dev-shm-usage')

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=C_options)

        try:
            driver.get(video_url)
            time.sleep(5)

            try:
                WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[text()='Reject all']"))
                ).click()
            except Exception:
                pass

            time.sleep(5)

            try:
                disclaimer_element = driver.find_element(By.XPATH, "//*[contains(text(), 'Includes paid promotion')]")
                return True if disclaimer_element else False
            except Exception:
                return False
        finally:
            driver.quit()

    def get_comments(self, video_id):
        self.progress.emit(f"Fetching comments for video {video_id}")
        comments = []
        
        try:
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100
            ).execute()

            for item in request['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            while 'nextPageToken' in request:
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,
                    pageToken=request['nextPageToken']
                ).execute()

                for item in request['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)

            return comments

        except HttpError as e:
            if e.resp.status == 403 and 'commentsDisabled' in str(e):
                return None
            else:
                raise

    def extract_channel_id(self, youtube_url):
        youtube_url = youtube_url.strip()

        match = re.match(r'(https?://)?(www\.)?youtube\.com/@([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            username = match.group(3)
            try:
                request = self.youtube.search().list(
                    part='snippet',
                    q=username,
                    type='channel',
                    maxResults=1
                ).execute()
        
                if request['items']:
                    return request['items'][0]['snippet']['channelId']
            except Exception as e:
                raise ValueError(f"Could not find channel ID for username: {username}")

        match = re.search(r'\"externalId\":\"(UC[\w-]+)\"', youtube_url)
        if match:
            return match.group(1)

        match = re.match(r'(https?://)?(www\.)?youtube\.com/channel/([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            return match.group(3)

        match = re.match(r'(https?://)?(www\.)?youtube\.com/user/([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            username = match.group(3)
            return self.get_channel_id_by_username(username)

        match = re.match(r'(https?://)?(www\.)?youtube\.com/c/([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            custom_name = match.group(3)
            return self.get_channel_id_by_custom_name(custom_name) 

        match = re.match(r'(https?://)?(www\.)?youtube\.com/([a-zA-Z0-9_-]+)', youtube_url)
        if match:
            homepage_name = match.group(3)
            return self.get_channel_id_by_custom_name(homepage_name)

        raise ValueError("Invalid YouTube URL format. Please enter a valid channel URL.")

    def get_channel_id_by_username(self, username):
        request = self.youtube.channels().list(part='id', forUsername=username).execute()
        if request['items']:
            return request['items'][0]['id']
        else:
            raise ValueError(f"Could not find a channel for username: {username}")

    def get_channel_id_by_custom_name(self, custom_name):
        request = self.youtube.search().list(part='snippet', q=custom_name, type='channel', maxResults=1).execute()
        if request['items']:
            return request['items'][0]['snippet']['channelId']
        else:
            raise ValueError(f"Could not find a channel for custom name: {custom_name}")
    
    def analyze_sentiment(self, comments):
        if not comments:
            return 0, {'Positive': 0, 'Negative': 0, 'Neutral': 0}

        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        scores = []

        for comment in comments:
            try:
                result = self.sentiment_pipeline(comment, truncation=True, max_length=512)[0]
                label = result['label']
                score = int(label.split()[0])  # Extract numeric score
            
                if score >= 4:
                   sentiment_counts['Positive'] += 1
                   normalized_score = 1.0
                elif score <= 2:
                   sentiment_counts['Negative'] += 1
                   normalized_score = -1.0
                else:
                   sentiment_counts['Neutral'] += 1
                   normalized_score = 0.0
            
                scores.append(normalized_score)
            except Exception as e:
                print(f"Error analyzing comment: {str(e)}")
                continue

        mean_sentiment = np.mean(scores) if scores else 0
        return round(mean_sentiment, 3), sentiment_counts
    
    def is_video_sponsored(self, video_title, video_description, video_url):
        sponsored_keywords = ['sponsored', 'paid promotion', 'partnered with', 'includes paid promotion', 'brand deal', 'paid partnership']
        combined_text = f"{video_title} {video_description}".lower()
        
        is_sponsored_by_keywords = any(keyword in combined_text for keyword in sponsored_keywords)
        is_sponsored_by_disclaimer = self.check_sponsorship_disclaimer(video_url)

        return is_sponsored_by_keywords or is_sponsored_by_disclaimer

    def get_video_description(self, video_id):
        request = self.youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()

        if request['items']:
            return request['items'][0]['snippet']['description']
        else:
            return None

    def get_video_engagement_metrics(self, video_id):
        try:
            request = self.youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()

            if 'items' in request and len(request['items']) > 0:
                stats = request['items'][0]['statistics']
                likes = int(stats.get('likeCount', 0))
                views = int(stats.get('viewCount', 0))
                comments_count = int(stats.get('commentCount', 0))
                return likes, views, comments_count
            else:
                return 0, 0, 0
        except Exception as e:
            self.error.emit(f"Error fetching engagement metrics for video {video_id}: {str(e)}")
            return 0, 0, 0

    def clean_text(self, text):

        text = text.lower()
        # Remove URLs
        text = regex.sub(r"http\S+|www\S+", "", text)
        # Remove mentions (@username)
        text = regex.sub(r"@\w+", "", text)
        # Remove hashtags (#example)
        text = regex.sub(r"#\w+", "", text)
        # Remove email addresses
        text = regex.sub(r"\S+@\S+", "", text)
        # Remove numbers (including patterns like "01 09")
        text = regex.sub(r"\b\d+(?:\s+\d+)*\b", "", text)
        # Remove emojis
        text = regex.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        # Using \p{L} to match any kind of letter from any language
        text = regex.sub(r"[^\p{L}\s]", "", text, flags=regex.UNICODE)
        # Normalize whitespace
        text = regex.sub(r'\s+', ' ', text)

        return text.strip()

    def get_channel_name(self, channel_id):
        request = self.youtube.channels().list(part='snippet', id=channel_id).execute()
        if request['items']:
            return request['items'][0]['snippet']['title']
        else:
            return None

    def evaluate_channel(self, channel_id, video_count, target_language='en'):
        videos = self.get_channel_videos(channel_id, video_count)
        sponsored_sentiments = []
        unsponsored_sentiments = []
        sponsored_sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        unsponsored_sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        sponsored_engagement_metrics = {"likes": 0, "views": 0, "comments": 0, "count": 0}
        unsponsored_engagement_metrics = {"likes": 0, "views": 0, "comments": 0, "count": 0}
        sponsored_videos = []
        unsponsored_videos = []

        for video in videos:
            self.progress.emit(f"Analyzing video: {video['video_title']}")
            video_id = video['video_id']
            video_title = video['video_title']
            video_url = video['video_url']
            video_description = self.get_video_description(video_id)

            comments = self.get_comments(video_id)
            if comments is None:
                continue

            comments = [self.clean_text(comment) for comment in comments]
            likes, views, comments_count = self.get_video_engagement_metrics(video_id)

            if self.is_video_sponsored(video_title, video_description, video_url):
                
                avg_sentiment, comment_sentiment_counts = self.analyze_sentiment(comments)
                sponsored_sentiments.append(avg_sentiment)

                sponsored_sentiment_counts['Positive'] += comment_sentiment_counts['Positive']
                sponsored_sentiment_counts['Negative'] += comment_sentiment_counts['Negative']
                sponsored_sentiment_counts['Neutral'] += comment_sentiment_counts['Neutral']

                sponsored_engagement_metrics['likes'] += likes
                sponsored_engagement_metrics['views'] += views
                sponsored_engagement_metrics['comments'] += comments_count
                sponsored_engagement_metrics['count'] += 1
                sponsored_videos.append(video)
            else:

                avg_sentiment, comment_sentiment_counts = self.analyze_sentiment(comments)
                unsponsored_sentiments.append(avg_sentiment)

                unsponsored_sentiment_counts['Positive'] += comment_sentiment_counts['Positive']
                unsponsored_sentiment_counts['Negative'] += comment_sentiment_counts['Negative']
                unsponsored_sentiment_counts['Neutral'] += comment_sentiment_counts['Neutral']

                unsponsored_engagement_metrics['likes'] += likes
                unsponsored_engagement_metrics['views'] += views
                unsponsored_engagement_metrics['comments'] += comments_count
                unsponsored_engagement_metrics['count'] += 1
                unsponsored_videos.append(video)

        num_sponsored = len(sponsored_videos)
        num_unsponsored = len(unsponsored_videos)

        avg_sponsored_sentiment = np.mean(sponsored_sentiments) if sponsored_sentiments else 0
        avg_unsponsored_sentiment = np.mean(unsponsored_sentiments) if unsponsored_sentiments else 0

        if sponsored_engagement_metrics['views'] > 0:
            avg_sponsored_engagement_score = (sponsored_engagement_metrics['likes'] + sponsored_engagement_metrics['comments']) / sponsored_engagement_metrics['views']
        else:
            avg_sponsored_engagement_score = 0

        if unsponsored_engagement_metrics['views'] > 0:
            avg_unsponsored_engagement_score = (unsponsored_engagement_metrics['likes'] + unsponsored_engagement_metrics['comments']) / unsponsored_engagement_metrics['views']
        else:
            avg_unsponsored_engagement_score = 0

        overall_sponsored_score = (0.7 * avg_sponsored_sentiment) + (0.3 * avg_sponsored_engagement_score)
        overall_unsponsored_score = (0.7 * avg_unsponsored_sentiment) + (0.3 * avg_unsponsored_engagement_score)

        return overall_sponsored_score, overall_unsponsored_score, num_sponsored, num_unsponsored, sponsored_sentiment_counts, unsponsored_sentiment_counts, avg_sponsored_engagement_score, avg_unsponsored_engagement_score, avg_sponsored_sentiment, avg_unsponsored_sentiment 


class ModernFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modernFrame")
        self.setStyleSheet("""
            QFrame#modernFrame {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)

class YouTubePartnerEstimator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Partner Estimator")
        self.setMinimumSize(1000, 700)

        if hasattr(sys, '_MEIPASS'):
            logo_path = os.path.join(sys._MEIPASS, 'Logo.png')
        else:
            logo_path = 'Logo.png'
        
        self.setWindowIcon(QIcon(logo_path))
     
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
            QLineEdit {
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #f11c87;
            }
            QPushButton#evaluateButton {
                background-color: #f11c87;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#evaluateButton:hover {
                background-color: #f00c68;
            }
            QPushButton#evaluateButton:disabled {
                background-color: #BDBDBD;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                gridline-color: #f5f5f5;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QProgressDialog {
                background-color: white;
                border-radius: 6px;
            }
            QProgressDialog QLabel {
                color: #333333;
                font-size: 14px;
            }
            QProgressBar {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                text-align: center;
                font-size: 14px;
                background-color: #f5f5f5;
                color: #333333;
            }
            QProgressBar::chunk {
                background-color: #f11c87; /* Pink gradient for modern look */
                border-radius: 8px;
            }
        """)
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        header_frame = ModernFrame()
        header_layout = QVBoxLayout(header_frame)
        
        title_label = QLabel("YouTube Partner Estimator")
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #f11c87;
            margin: 20px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        
        subtitle_label = QLabel("Analyze YouTube channels for partnership potential")
        subtitle_label.setStyleSheet("""
            font-size: 16px;
            color: #757575;
        """)
        subtitle_label.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        main_layout.addWidget(header_frame)

        # Input section
        input_frame = ModernFrame()
        input_layout = QHBoxLayout(input_frame)
        input_layout.setSpacing(15)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube Channel URL")
        self.url_input.setStyleSheet("""
            QLineEdit {
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: white;
                font-size: 14px;
                min-width: 300px;
            }
            QLineEdit:focus {
                border: 2px solid #f11c87;
            }
        """)


        self.evaluate_button = QPushButton("Evaluate Channel")
        self.evaluate_button.setObjectName("evaluateButton")
        self.evaluate_button.setFixedWidth(200)
        self.evaluate_button.clicked.connect(self.start_evaluation)

        self.video_count_spinner = QSpinBox()
        self.video_count_spinner.setRange(2, 50)
        self.video_count_spinner.setValue(4)
        self.video_count_spinner.setStyleSheet("""
            QSpinBox {
                padding: 12px;
                border: 2px solid #e0e0e0;
                
                border-radius: 6px;
                background-color: white;
                font-size: 14px;
                min-width: 80px;
            }
            QSpinBox:focus {
                border: 2px solid #f11c87;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                background-color: transparent;
                border: none;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #f5f5f5;
            }
            QSpinBox::up-arrow {
                image: url(arrow.svg);
                width: 10px;
                height: 10px;
            }
            QSpinBox::down-arrow {
                image: url(down.svg);
                width: 10px;
                height: 10px;
            }
        """)
        
        input_layout.addWidget(self.url_input)
        input_layout.addWidget(self.video_count_spinner)
        input_layout.addWidget(self.evaluate_button)

        main_layout.addWidget(input_frame)

        # Results section
        results_frame = ModernFrame()
        results_layout = QVBoxLayout(results_frame)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setShowGrid(False)
        
        results_layout.addWidget(self.results_table)
        main_layout.addWidget(results_frame)

        # Recommendation section
        recommendation_frame = ModernFrame()
        recommendation_layout = QVBoxLayout(recommendation_frame)
        
        self.recommendation_label = QLabel()
        self.recommendation_label.setAlignment(Qt.AlignCenter)
        self.recommendation_label.setStyleSheet("""
            font-size: 16px;
            padding: 20px;
            border-radius: 6px;
        """)
        
        recommendation_layout.addWidget(self.recommendation_label)
        main_layout.addWidget(recommendation_frame)

        # Set stretch factors
        main_layout.setStretch(0, 0)  # Header
        main_layout.setStretch(1, 0)  # Input
        main_layout.setStretch(2, 1)  # Results
        main_layout.setStretch(3, 0)  # Recommendation


    def start_evaluation(self):
        channel_url = self.url_input.text().strip()
        if not channel_url:
            self.show_error_message("Please enter a YouTube channel URL")
            return

        video_count = self.video_count_spinner.value()

        self.evaluate_button.setEnabled(False)
        self.worker = EvaluationWorker(channel_url, video_count)
        self.worker.finished.connect(self.handle_results)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

        self.progress_dialog = QProgressDialog("Evaluating channel...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Analysis in Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setStyleSheet("""
            QProgressDialog {
                background-color: white;
                border-radius: 10px;
                min-width: 400px;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
        """)
        self.progress_dialog.show()

    def show_error_message(self, message):
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
        """)
        error_dialog.exec_()

    def handle_error(self, error_message):
        self.progress_dialog.hide()
        self.evaluate_button.setEnabled(True)
        self.show_error_message(error_message)

    def update_progress(self, message):
        self.progress_dialog.setLabelText(message)

    def handle_results(self, results):
        self.progress_dialog.hide()
        self.evaluate_button.setEnabled(True)

        # Clear and update results table
        self.results_table.setRowCount(0)
        metrics = [
            ('Channel Name', results['channel_name']),
            ('Channel ID', results['channel_id']),
            ('Sponsored Content Score', f"{results['sponsored_score']:.4f}"),
            ('Organic Content Score', f"{results['unsponsored_score']:.4f}"),
            ('Sponsored Videos Analyzed', str(results['num_sponsored'])),
            ('Organic Videos Analyzed', str(results['num_unsponsored'])),
            ('Sponsored Sentiment Score', str(results['Sponsored_Sentiment_Score'])),
            ('Unsponsored Sentiment Score',str(results['Unsponsored_Sentiment_Score'])),
            ('Sponsored Engagement Score', str(results['Sponsored_Engagement_Score'])),
            ('Unsponsored Engagement Score', str(results['Unsponsored_Engagement_Score'])),
            ('Sponsored Positive Comments', str(results['sponsored_positive_comments'])),
            ('Sponsored Negative Comments', str(results['sponsored_negative_comments'])),
            ('Sponsored Neutral Comments', str(results['sponsored_neutral_comments'])),
            ('Organic Positive Comments', str(results['unsponsored_positive_comments'])),
            ('Organic Negative Comments', str(results['unsponsored_negative_comments'])),
            ('Organic Neutral Comments', str(results['unsponsored_neutral_comments']))
        ]

        self.results_table.setRowCount(len(metrics))
        for i, (metric, value) in enumerate(metrics):
            self.results_table.setItem(i, 0, QTableWidgetItem(metric))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Update recommendation with modern styling
        if results['sponsored_score'] > results['unsponsored_score']:
            self.recommendation_label.setText("✨ This channel shows strong potential for sponsored partnerships! ✨")
            self.recommendation_label.setStyleSheet("""
                background-color: #E8F5E9;
                color: #2E7D32;
                font-size: 16px;
                font-weight: bold;
                padding: 20px;
                border-radius: 6px;
            """)  
        else:
            self.recommendation_label.setText("⚠️ This channel may need improvement before pursuing sponsored partnerships")
            self.recommendation_label.setStyleSheet("""
                background-color: #FFEBEE;
                color: #C62828;
                font-size: 16px;
                font-weight: bold;
                padding: 20px;
                border-radius: 6px;
            """)

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = YouTubePartnerEstimator()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

