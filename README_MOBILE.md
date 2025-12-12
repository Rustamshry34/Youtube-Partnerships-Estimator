# YouTube Partner Estimator - Mobile Version

This is a mobile-optimized version of the YouTube Partner Estimator application, converted from the original PyQt desktop application to Kivy for cross-platform mobile compatibility.

## Features

- Modern, responsive UI designed for mobile devices
- Channel analysis functionality for YouTube partnership potential
- Sentiment analysis of comments using BERT
- Engagement metrics for sponsored vs. organic content
- Partnership recommendation system
- Mobile-adapted architecture with threading for smooth UI

## Key Differences from Desktop Version

1. **UI Framework**: Converted from PyQt to Kivy for mobile compatibility
2. **Selenium Limitation**: Removed Selenium-based disclaimer checking (not supported on mobile)
3. **Responsive Design**: Optimized for various screen sizes and orientations
4. **Touch-First Interface**: Designed for touch interactions

## Requirements

- Python 3.7+
- Kivy 2.2.1
- Google API Python Client
- Transformers library
- PyTorch
- Other dependencies listed in requirements_mobile.txt

## Installation

1. Install dependencies:
```bash
pip install -r requirements_mobile.txt
```

2. Replace the placeholder API key in the code with your actual YouTube Data API key

3. Run the application:
```bash
python YoutubePartnerEstimator_Mobile.py
```

## Building for Mobile

To build for Android using Buildozer:
1. Install Buildozer: `pip install buildozer`
2. Initialize buildozer: `buildozer init`
3. Modify the buildozer.spec file to include required permissions and dependencies
4. Build: `buildozer android debug`

For iOS, you can use Kivy-iOS following similar steps.

## API Key Setup

1. Create a Google Cloud Project
2. Enable the YouTube Data API v3
3. Create an API key
4. Replace the placeholder API key in the source code with your actual key

## Limitations

- Selenium-based functionality has been removed due to mobile platform restrictions
- Some advanced UI effects may be simplified for mobile performance
- File system access may be limited depending on the target platform

## Architecture

The mobile version maintains the core functionality of the original application:
- Channel URL parsing and validation
- Video fetching and analysis
- Sentiment analysis using BERT
- Engagement metrics calculation
- Partnership recommendation algorithm

## Contributing

Feel free to submit issues and enhancement requests via the GitHub repository.