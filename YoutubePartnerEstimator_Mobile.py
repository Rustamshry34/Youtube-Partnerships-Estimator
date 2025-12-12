from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.utils import get_color_from_hex
from kivy.properties import NumericProperty
from kivy.lang import Builder
from kivy.uix.widget import Widget
import sys
import threading
import time
from kivy.network.urlrequest import UrlRequest
import json
import math

# Import necessary modules for the functionality (these would need to be adapted for mobile)
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from transformers import pipeline
    import torch
    import isodate
    import numpy as np
    import regex
    import re
    import os
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install required packages for full functionality")


class MobileEvaluationWorker:
    """
    Adaptation of the original EvaluationWorker class for mobile use.
    Some functionality may be limited due to mobile platform restrictions.
    """
    def __init__(self, channel_url, video_count, api_key):
        self.channel_url = channel_url
        self.video_count = video_count 
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.sentiment_pipeline = None
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
                print(f"Error: Model directory not found at {model_path}")
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

    def run(self):
        try:
            channel_id = self.extract_channel_id(self.channel_url)
            channel_name = self.get_channel_name(channel_id)

            print("Evaluating channel...")
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
            
            return results
        except Exception as e:
            raise e

    def get_channel_videos(self, channel_id, video_count):
        print("Fetching channel videos...")
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
        """
        On mobile, we might not be able to use Selenium effectively.
        This method would need to be adapted for mobile platforms.
        For now, returning False to indicate no sponsorship detected.
        """
        print(f"Checking sponsorship disclaimer for {video_url}")
        # Mobile adaptation: return False since we can't use Selenium on mobile
        return False

    def get_comments(self, video_id):
        print(f"Fetching comments for video {video_id}")
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
        """
        On mobile, we can't use Selenium to check for disclaimers, 
        so we rely only on keywords in title and description.
        """
        sponsored_keywords = ['sponsored', 'paid promotion', 'partnered with', 'includes paid promotion', 'brand deal', 'paid partnership']
        combined_text = f"{video_title} {video_description}".lower()
        
        is_sponsored_by_keywords = any(keyword in combined_text for keyword in sponsored_keywords)
        # Mobile adaptation: skip Selenium-based disclaimer checking
        # is_sponsored_by_disclaimer = self.check_sponsorship_disclaimer(video_url)

        return is_sponsored_by_keywords # or is_sponsored_by_disclaimer

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
            print(f"Error fetching engagement metrics for video {video_id}: {str(e)}")
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
            print(f"Analyzing video: {video['video_title']}")
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

# Define custom styles
Builder.load_string('''
<ModernLabel@Label>:
    color: 0.2, 0.2, 0.2, 1
    font_size: '16sp'
    halign: 'left'

<ModernTextInput@TextInput>:
    multiline: False
    background_color: 1, 1, 1, 1
    foreground_color: 0.2, 0.2, 0.2, 1
    hint_text_color: 0.6, 0.6, 0.6, 1
    cursor_color: 0.16, 0.2, 0.74, 1
    size_hint_y: None
    height: '40dp'
    padding: '12dp'
    font_size: '16sp'

<ModernButton@Button>:
    background_normal: ''
    background_color: 0.16, 0.2, 0.74, 1
    color: 1, 1, 1, 1
    font_size: '16sp'
    size_hint_y: None
    height: '50dp'
    bold: True

<ModernSpinner@Spinner>:
    background_normal: ''
    background_color: 1, 1, 1, 1
    color: 0.2, 0.2, 0.2, 1
    size_hint_y: None
    height: '40dp'
    font_size: '16sp'

<ResultCard@BoxLayout>:
    orientation: 'vertical'
    padding: '15dp'
    spacing: '10dp'
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size
        Color:
            rgba: 0.9, 0.9, 0.9, 1
        Rectangle:
            pos: self.x + dp(2), self.y + dp(2)
            size: self.width - dp(4), self.height - dp(4)
''')

class CircularProgressWidget(Widget):
    progress = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super(CircularProgressWidget, self).__init__(**kwargs)
        self.bind(pos=self.update_graphics, size=self.update_graphics)
        
    def update_graphics(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(0.9, 0.9, 0.9, 1)
            Rectangle(pos=self.pos, size=self.size)
            
            Color(0.16, 0.2, 0.74, 1)
            # Draw arc based on progress
            angle_start = 0
            angle_end = self.progress * 3.6  # Convert percentage to degrees
            # Draw a simple circular progress indicator
            # In a real implementation, this would draw an actual arc

class MobileEstimatorApp(App):
    def build(self):
        # Set window background color
        Window.clearcolor = get_color_from_hex('#f0f2fa')
        
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding='20dp', spacing='20dp')
        
        # Header section
        header_layout = BoxLayout(size_hint_y=None, height='100dp')
        header_layout.canvas.before:
            with header_layout.canvas:
                Color(0.16, 0.2, 0.74, 1)  # Blue gradient start
                Rectangle(pos=header_layout.pos, size=header_layout.size)
        
        title_label = Label(
            text='YouTube Partner Estimator',
            color=(1, 1, 1, 1),
            font_size='24sp',
            bold=True
        )
        subtitle_label = Label(
            text='Analyze YouTube channels for partnership potential',
            color=(0.9, 0.9, 0.9, 1),
            font_size='14sp'
        )
        
        header_content = BoxLayout(orientation='vertical', padding='15dp')
        header_content.add_widget(title_label)
        header_content.add_widget(subtitle_label)
        header_layout.add_widget(header_content)
        
        # Input section
        input_card = BoxLayout(orientation='vertical', size_hint_y=None, height='200dp', padding='15dp', spacing='15dp')
        input_card.canvas.before:
            with input_card.canvas:
                Color(1, 1, 1, 1)
                Rectangle(pos=input_card.pos, size=input_card.size)
        
        # URL input
        url_label = Label(text='Channel URL:', size_hint_y=None, height='30dp', halign='left')
        self.url_input = TextInput(hint_text='Enter YouTube Channel URL', multiline=False)
        
        # Video count spinner
        count_layout = BoxLayout(size_hint_y=None, height='40dp', spacing='10dp')
        count_label = Label(text='Videos to analyze:', size_hint_x=0.6)
        self.video_count_spinner = Spinner(
            text='4',
            values=['2', '4', '6', '8', '10', '12', '15', '20'],
            size_hint_x=0.4
        )
        count_layout.add_widget(count_label)
        count_layout.add_widget(self.video_count_spinner)
        
        # Evaluate button
        self.evaluate_button = Button(text='Evaluate Channel', on_press=self.start_evaluation)
        
        input_card.add_widget(url_label)
        input_card.add_widget(self.url_input)
        input_card.add_widget(count_layout)
        input_card.add_widget(self.evaluate_button)
        
        # Results section (initially empty)
        results_scroll = ScrollView(size_hint_y=1)
        self.results_layout = GridLayout(cols=1, spacing='10dp', size_hint_y=None)
        self.results_layout.bind(minimum_height=self.results_layout.setter('height'))
        
        results_label = Label(
            text='Analysis Results will appear here',
            size_hint_y=None,
            height='40dp',
            color=(0.2, 0.2, 0.2, 1),
            font_size='18sp',
            bold=True
        )
        self.results_layout.add_widget(results_label)
        
        results_scroll.add_widget(self.results_layout)
        
        # Add all sections to main layout
        main_layout.add_widget(header_layout)
        main_layout.add_widget(input_card)
        main_layout.add_widget(results_scroll)
        
        return main_layout
    
    def start_evaluation(self, instance):
        """Start the evaluation process"""
        channel_url = self.url_input.text.strip()
        if not channel_url:
            self.show_error_popup("Please enter a YouTube channel URL")
            return
        
        video_count = int(self.video_count_spinner.text)
        
        # Disable button during processing
        self.evaluate_button.disabled = True
        self.evaluate_button.text = "Processing..."
        
        # Start evaluation in a separate thread
        Clock.schedule_once(lambda dt: self.run_evaluation_in_thread(channel_url, video_count))
    
    def run_evaluation_in_thread(self, channel_url, video_count):
        """Run the evaluation in a separate thread to avoid blocking the UI"""
        def eval_func():
            try:
                # Create a worker with a placeholder API key
                # In a real implementation, you would use your actual YouTube API key
                api_key = 'XXXXXXXXXXXXXXXXXXXXXXXXX'  # Replace with your actual API key
                worker = MobileEvaluationWorker(channel_url, video_count, api_key)
                
                # Run the evaluation
                results = worker.run()
                
                # Schedule UI updates back on the main thread
                Clock.schedule_once(lambda dt: self.display_results(results))
            except Exception as e:
                Clock.schedule_once(lambda dt: self.handle_error(str(e)))
        
        thread = threading.Thread(target=eval_func)
        thread.daemon = True
        thread.start()
    
    def display_results(self, results):
        """Display the results in the UI"""
        # Clear previous results
        self.results_layout.clear_widgets()
        
        # Add title
        results_title = Label(
            text='Analysis Results',
            size_hint_y=None,
            height='40dp',
            color=(0.16, 0.2, 0.74, 1),
            font_size='20sp',
            bold=True
        )
        self.results_layout.add_widget(results_title)
        
        # Add results in organized sections
        self.add_result_section("Channel Information", [
            ('Channel Name', results['channel_name']),
            ('Channel ID', results['channel_id'])
        ])
        
        self.add_result_section("Performance Scores", [
            ('Sponsored Content Score', f"{results['sponsored_score']:.4f}"),
            ('Organic Content Score', f"{results['unsponsored_score']:.4f}")
        ])
        
        self.add_result_section("Video Analysis", [
            ('Sponsored Videos Analyzed', str(results['num_sponsored'])),
            ('Organic Videos Analyzed', str(results['num_unsponsored']))
        ])
        
        self.add_result_section("Sentiment Analysis", [
            ('Sponsored Sentiment Score', f"{results['Sponsored_Sentiment_Score']:.4f}"),
            ('Organic Sentiment Score', f"{results['Unsponsored_Sentiment_Score']:.4f}")
        ])
        
        self.add_result_section("Engagement Metrics", [
            ('Sponsored Engagement Score', f"{results['Sponsored_Engagement_Score']:.4f}"),
            ('Organic Engagement Score', f"{results['Unsponsored_Engagement_Score']:.4f}")
        ])
        
        self.add_result_section("Comment Analysis (Sponsored)", [
            ('Positive Comments', str(results['sponsored_positive_comments'])),
            ('Negative Comments', str(results['sponsored_negative_comments'])),
            ('Neutral Comments', str(results['sponsored_neutral_comments']))
        ])
        
        self.add_result_section("Comment Analysis (Organic)", [
            ('Positive Comments', str(results['unsponsored_positive_comments'])),
            ('Negative Comments', str(results['unsponsored_negative_comments'])),
            ('Neutral Comments', str(results['unsponsored_neutral_comments']))
        ])
        
        # Add recommendation
        rec_layout = BoxLayout(orientation='vertical', size_hint_y=None, height='100dp', padding='15dp', spacing='10dp')
        rec_layout.canvas.before:
            with rec_layout.canvas:
                Color(0.9, 0.95, 0.9, 1) if results['sponsored_score'] > results['unsponsored_score'] else Color(1.0, 0.98, 0.85, 1)
                Rectangle(pos=rec_layout.pos, size=rec_layout.size)
        
        if results['sponsored_score'] > results['unsponsored_score']:
            rec_text = "✨ This channel shows strong potential for sponsored partnerships! The sponsored content performs well with positive audience engagement and sentiment."
            rec_color = (0.18, 0.49, 0.19, 1)  # Green
        else:
            rec_text = "⚠️ This channel may need improvement before pursuing sponsored partnerships. The organic content currently outperforms sponsored content in terms of engagement and sentiment."
            rec_color = (0.95, 0.65, 0.15, 1)  # Orange
        
        recommendation_label = Label(
            text=rec_text,
            color=rec_color,
            font_size='16sp',
            halign='center',
            valign='middle'
        )
        
        rec_layout.add_widget(Label(text='Partnership Recommendation', color=(0.16, 0.2, 0.74, 1), bold=True))
        rec_layout.add_widget(recommendation_label)
        self.results_layout.add_widget(rec_layout)
        
        # Re-enable the button
        self.evaluate_button.disabled = False
        self.evaluate_button.text = "Evaluate Channel"
    
    def add_result_section(self, title, metrics):
        """Add a section of results to the layout"""
        section_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(40 + len(metrics) * 30), padding='10dp', spacing='5dp')
        section_layout.canvas.before:
            with section_layout.canvas:
                Color(0.95, 0.95, 1, 1)
                Rectangle(pos=section_layout.pos, size=section_layout.size)
        
        # Section title
        title_label = Label(
            text=title,
            size_hint_y=None,
            height='30dp',
            color=(0.16, 0.2, 0.74, 1),
            font_size='16sp',
            bold=True
        )
        section_layout.add_widget(title_label)
        
        # Add each metric
        for metric, value in metrics:
            metric_row = BoxLayout(size_hint_y=None, height='30dp', spacing='10dp')
            
            metric_label = Label(
                text=f"[b]{metric}[/b]:",
                markup=True,
                size_hint_x=0.6,
                halign='left'
            )
            value_label = Label(
                text=str(value),
                size_hint_x=0.4,
                halign='right'
            )
            
            metric_row.add_widget(metric_label)
            metric_row.add_widget(value_label)
            section_layout.add_widget(metric_row)
        
        self.results_layout.add_widget(section_layout)
    
    def handle_error(self, error_message):
        """Handle errors during evaluation"""
        self.show_error_popup(error_message)
        self.evaluate_button.disabled = False
        self.evaluate_button.text = "Evaluate Channel"
    
    def show_error_popup(self, message):
        """Show an error popup"""
        popup_layout = BoxLayout(orientation='vertical', padding='20dp', spacing='20dp')
        
        error_label = Label(text=message, text_size=(Window.width * 0.8, None), halign='center')
        close_button = Button(text='Close', size_hint_y=None, height='40dp')
        
        popup_layout.add_widget(error_label)
        popup_layout.add_widget(close_button)
        
        popup = Popup(title='Error', content=popup_layout, size_hint=(0.8, 0.4))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

if __name__ == '__main__':
    MobileEstimatorApp().run()