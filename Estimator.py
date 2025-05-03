import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QLabel, QLineEdit, QPushButton, QTableWidget,
                           QTableWidgetItem, QMessageBox,
                           QFrame, QHBoxLayout, QSpinBox,
                           QHeaderView, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QColor, QBrush, QPainter, QPen
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import pipeline
import time
import os 
import regex
import torch
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
        self.api_key = 'XXXXXXXXXXXXXXXXXXXXXXXXX'
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.setup_nltk()


    def setup_nltk(self):
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        
        try:
            if hasattr(sys, '_MEIPASS'):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
        
            model_path = os.path.join(base_path, "model_dir")
            
            if not os.path.exists(model_path):
                self.error.emit(f"Error: Model directory not found at {model_path}")
                return

            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_path,
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
        

        #if hasattr(sys, '_MEIPASS'):
            #base_path = sys._MEIPASS 
        #else:
            #base_path = os.path.dirname(os.path.abspath(__file__))

        #chromedriver_path = os.path.join(base_path, 'chromedriver.exe')
        #chrome_path = os.path.join(base_path, 'chrome-portable', 'chrome.exe')

        #service = Service(chromedriver_path)
        #C_options.binary_location = chrome_path 

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
                # Truncate long comments to max token length
                #tokens = self.tokenizer.encode(comment, truncation=True, max_length=512)
                # BERT model returns scores from 1 to 5
                result = self.sentiment_pipeline(comment, truncation=True, max_length=512)[0]
                # Convert 1-5 score to sentiment category
                label = result['label']
                score = int(label.split()[0])  # Extract numeric score
            
                # Convert 5-point scale to sentiment categories
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
        # Remove any characters not belonging to letters or whitespace
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

    def evaluate_channel(self, channel_id, video_count):
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
    def __init__(self, parent=None, shadow_strength=3):
        super().__init__(parent)
        self.setObjectName("modernFrame")
        self.setStyleSheet("""
            QFrame#modernFrame {
                background-color: white;
                border-radius: 12px;
                border: none;
            }
        """)
        
        # Add drop shadow effect
        if shadow_strength > 0:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(0, 0, 0, 25))
            shadow.setOffset(0, 3)
            self.setGraphicsEffect(shadow)


# Circular progress indicator (for visual flair during loading)
class CircularProgressIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.angle = 0
        self.timer = self.startTimer(30)
        
    def timerEvent(self, event):
        self.angle = (self.angle + 5) % 360
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen()
        pen.setWidth(3)
        pen.setColor(QColor("#2834bd"))
        painter.setPen(pen)
        
        rect = self.rect().adjusted(5, 5, -5, -5)
        painter.drawArc(rect, self.angle * 16, 120 * 16)


# Modern, flat-styled progress dialog
class ModernProgressDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(400, 180)
        
        layout = QVBoxLayout(self)
        
        # Content frame with shadow
        frame = ModernFrame(self)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(30, 30, 30, 30)
        
        # Progress indicator and message in horizontal layout
        progress_layout = QHBoxLayout()
        self.indicator = CircularProgressIndicator()
        self.message_label = QLabel("Evaluating channel...")
        self.message_label.setStyleSheet("font-size: 14px; color: #333333;")
        
        progress_layout.addWidget(self.indicator, 0, Qt.AlignCenter)
        progress_layout.addWidget(self.message_label, 1, Qt.AlignCenter)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #333333;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        
        frame_layout.addLayout(progress_layout)
        frame_layout.addWidget(self.cancel_button, 0, Qt.AlignCenter)
        
        layout.addWidget(frame)
        
    def set_message(self, message):
        self.message_label.setText(message)
        
    def center_on_parent(self, parent):
        if parent:
            parent_geo = parent.geometry()
            x = parent_geo.x() + (parent_geo.width() - self.width()) // 2
            y = parent_geo.y() + (parent_geo.height() - self.height()) // 2
            self.move(x, y)


# Modern styled QLineEdit with built-in clear button
class ModernLineEdit(QLineEdit):
    def __init__(self, placeholder_text="", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder_text)
        self.setStyleSheet("""
            QLineEdit {
                padding: 12px 12px 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
                font-size: 14px;
                min-width: 300px;
                min-height: 24px;
            }
            QLineEdit:focus {
                border: 2px solid #2834bd;
            }
        """)
        
        # Add clear button
        self.setClearButtonEnabled(True)


# Main application window with redesigned UI
class YouTubePartnerEstimator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Partner Estimator")
        self.setMinimumSize(1200, 900)

        # Load logo
        if hasattr(sys, '_MEIPASS'):
            logo_path = os.path.join(sys._MEIPASS, 'MLogo.png')
        else:
            logo_path = 'MLogo.png'
        
        self.setWindowIcon(QIcon(logo_path))
        
        # Set the application-wide stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f7f8fc;
            }
            QLabel {
                color: #333333;
            }
            QPushButton#evaluateButton {
                background-color: #2834bd;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                min-height: 24px;
            }
            QPushButton#evaluateButton:hover {
                background-color: #3a46d8;
            }
            QPushButton#evaluateButton:pressed {
                background-color: #1a26a5;
            }
            QPushButton#evaluateButton:disabled {
                background-color: #BDBDBD;
            }
            QTableWidget {
                background-color: white;
                border: none;
                border-radius: 8px;
                gridline-color: #f5f5f5;
                selection-background-color: #e8eeff;
                selection-color: #2834bd;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f0f0f0;
            }
            QTableWidget::item:selected {
                background-color: #e8eeff;
                color: #2834bd;
            }
            QHeaderView::section {
                background-color: #f0f2fa;
                padding: 12px;
                border: none;
                font-weight: bold;
                color: #2834bd;
            }
            QSpinBox {
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
                font-size: 14px;
                min-width: 80px;
                min-height: 24px;
            }
            QSpinBox:focus {
                border: 2px solid #2834bd;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                background-color: transparent;
                border: none;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #f5f5f5;
            }
        """)
        
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Header section with gradient background
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_frame.setStyleSheet("""
            QFrame#headerFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2834bd, stop:1 #4f57d6);
                border-radius: 12px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(30, 25, 30, 25)
        
        title_label = QLabel("YouTube Partner Estimator")
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: white;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        
        subtitle_label = QLabel("Analyze YouTube channels for partnership potential")
        subtitle_label.setStyleSheet("""
            font-size: 16px;
            color: rgba(255, 255, 255, 0.9);
        """)
        subtitle_label.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        main_layout.addWidget(header_frame)

        # Input section with improved styling
        input_frame = ModernFrame()
        input_layout = QHBoxLayout(input_frame)
        input_layout.setSpacing(15)
        input_layout.setContentsMargins(30, 30, 30, 30)

        # Channel URL input with label
        url_label = QLabel("Channel URL:")
        url_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.url_input = ModernLineEdit("Enter YouTube Channel URL")
        
        # Video count selection with label
        count_label = QLabel("Videos to analyze:")
        count_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.video_count_spinner = QSpinBox()
        self.video_count_spinner.setRange(2, 50)
        self.video_count_spinner.setValue(4)
        self.video_count_spinner.setStyleSheet("""
            QSpinBox {
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
                font-size: 14px;
            }
        """)

        # Evaluate button
        self.evaluate_button = QPushButton("Evaluate Channel")
        self.evaluate_button.setObjectName("evaluateButton")
        self.evaluate_button.setFixedWidth(200)
        self.evaluate_button.clicked.connect(self.start_evaluation)

        # Add all components to the same horizontal layout
        input_layout.addWidget(url_label)
        input_layout.addWidget(self.url_input, 1)  # Give URL input more stretch
        input_layout.addWidget(count_label)
        input_layout.addWidget(self.video_count_spinner)
        input_layout.addWidget(self.evaluate_button)

        main_layout.addWidget(input_frame)

        # Results section with tabs for different data views
        results_frame = ModernFrame()
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(30, 30, 30, 30)
        
        # Results heading
        results_header = QLabel("Analysis Results")
        results_header.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2834bd;
            margin-bottom: 15px;
        """)
        results_layout.addWidget(results_header)
        
        # Results table with improved styling
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        
        # Configure table appearance
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setShowGrid(False)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        results_layout.addWidget(self.results_table)
        main_layout.addWidget(results_frame, 1)  # Give this widget more stretch

        # Recommendation section with card-like appearance
        recommendation_frame = ModernFrame()
        recommendation_layout = QVBoxLayout(recommendation_frame)
        recommendation_layout.setContentsMargins(30, 25, 30, 25)
        
        recommendation_header = QLabel("Partnership Recommendation")
        recommendation_header.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #2834bd;
        """)
        
        self.recommendation_label = QLabel("Enter a YouTube channel URL and click 'Evaluate Channel' to get a recommendation")
        self.recommendation_label.setAlignment(Qt.AlignCenter)
        self.recommendation_label.setStyleSheet("""
            font-size: 15px;
            padding: 20px;
            border-radius: 8px;
            background-color: #f0f2fa;
            color: #666666;
        """)
        self.recommendation_label.setWordWrap(True)
        
        recommendation_layout.addWidget(recommendation_header)
        recommendation_layout.addWidget(self.recommendation_label)
        main_layout.addWidget(recommendation_frame)

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

        # Create and show modern progress dialog
        self.progress_dialog = ModernProgressDialog(self)
        self.progress_dialog.cancel_button.clicked.connect(self.cancel_evaluation)
        self.progress_dialog.center_on_parent(self)
        self.progress_dialog.show()

    def cancel_evaluation(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.evaluate_button.setEnabled(True)
            self.progress_dialog.close()

    def show_error_message(self, message):
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.setStyleSheet("""
            QMessageBox {
                background-color: white;
                border-radius: 8px;
            }
            QLabel {
                color: #333333;
                min-width: 300px;
            }
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        error_dialog.exec_()

    def handle_error(self, error_message):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        self.evaluate_button.setEnabled(True)
        self.show_error_message(error_message)

    def update_progress(self, message):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.set_message(message)

    def handle_results(self, results):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        self.evaluate_button.setEnabled(True)

        # Clear and update results table
        self.results_table.setRowCount(0)
        
        # Group metrics for better organization
        metrics_groups = {
            "Channel Information": [
                ('Channel Name', results['channel_name']),
                ('Channel ID', results['channel_id'])
            ],
            "Performance Scores": [
                ('Sponsored Content Score', f"{results['sponsored_score']:.4f}"),
                ('Organic Content Score', f"{results['unsponsored_score']:.4f}")
            ],
            "Video Analysis": [
                ('Sponsored Videos Analyzed', str(results['num_sponsored'])),
                ('Organic Videos Analyzed', str(results['num_unsponsored']))
            ],
            "Sentiment Analysis": [
                ('Sponsored Sentiment Score', f"{results['Sponsored_Sentiment_Score']:.4f}"),
                ('Organic Sentiment Score', f"{results['Unsponsored_Sentiment_Score']:.4f}")
            ],
            "Engagement Metrics": [
                ('Sponsored Engagement Score', f"{results['Sponsored_Engagement_Score']:.4f}"),
                ('Organic Engagement Score', f"{results['Unsponsored_Engagement_Score']:.4f}")
            ],
            "Comment Analysis (Sponsored)": [
                ('Positive Comments', str(results['sponsored_positive_comments'])),
                ('Negative Comments', str(results['sponsored_negative_comments'])),
                ('Neutral Comments', str(results['sponsored_neutral_comments']))
            ],
            "Comment Analysis (Organic)": [
                ('Positive Comments', str(results['unsponsored_positive_comments'])),
                ('Negative Comments', str(results['unsponsored_negative_comments'])),
                ('Neutral Comments', str(results['unsponsored_neutral_comments']))
            ]
        }
        
        # Flatten the grouped metrics for display
        all_metrics = []
        for group, metrics in metrics_groups.items():
            all_metrics.append((f"--- {group} ---", ""))
            all_metrics.extend(metrics)
        
        # Populate the table
        self.results_table.setRowCount(len(all_metrics))
        for i, (metric, value) in enumerate(all_metrics):
            metric_item = QTableWidgetItem(metric)
            value_item = QTableWidgetItem(str(value))
            
            # Style group headers differently
            if metric.startswith("---"):
                metric_item.setBackground(QColor("#f0f2fa"))
                metric_item.setForeground(QBrush(QColor("#2834bd")))
                metric_item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                value_item.setBackground(QColor("#f0f2fa"))
            
            self.results_table.setItem(i, 0, metric_item)
            self.results_table.setItem(i, 1, value_item)

        # Update recommendation with modern styling
        if results['sponsored_score'] > results['unsponsored_score']:
            self.recommendation_label.setText("✨ This channel shows strong potential for sponsored partnerships! The sponsored content performs well with positive audience engagement and sentiment.")
            self.recommendation_label.setStyleSheet("""
                background-color: #e8f5e9;
                color: #2e7d32;
                font-size: 15px;
                font-weight: 500;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #4caf50;
            """)  
        else:
            self.recommendation_label.setText("⚠️ This channel may need improvement before pursuing sponsored partnerships. The organic content currently outperforms sponsored content in terms of engagement and sentiment.")
            self.recommendation_label.setStyleSheet("""
                background-color: #fff8e1;
                color: #f57c00;
                font-size: 15px;
                font-weight: 500;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #ff9800;
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

