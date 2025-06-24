import requests
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class NewsFetcher:
    def __init__(self):
        self.rss_feeds = {
            'crypto': [
                'https://cointelegraph.com/rss',
                'https://feeds.feedburner.com/coindesk/CoinDesk',
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://u.today/rss',
                'https://cryptopotato.com/feed/'
            ],
            'financial': [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://www.marketwatch.com/rss/topstories',
                'https://www.reuters.com/business/finance/rss',
                'https://feeds.bloomberg.com/markets/news.rss'
            ]
        }
        
        self.crypto_keywords = [
            'bitcoin', 'btc', 'cryptocurrency', 'crypto', 'blockchain',
            'ethereum', 'eth', 'defi', 'trading', 'market', 'price',
            'bull', 'bear', 'pump', 'dump', 'rally', 'crash'
        ]
        
    def fetch_crypto_news(self, hours_back=24):
        """Fetch recent cryptocurrency news from multiple sources"""
        all_news = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for category, feeds in self.rss_feeds.items():
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:10]:  # Limit to recent entries
                        # Parse published date
                        try:
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                pub_date = datetime(*entry.published_parsed[:6])
                            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                                pub_date = datetime(*entry.updated_parsed[:6])
                            else:
                                pub_date = datetime.now()
                        except:
                            pub_date = datetime.now()
                        
                        # Only include recent news
                        if pub_date >= cutoff_time:
                            # Check if crypto-related
                            title = entry.get('title', '').lower()
                            summary = entry.get('summary', '').lower()
                            content = f"{title} {summary}"
                            
                            is_crypto_related = any(keyword in content for keyword in self.crypto_keywords)
                            
                            if is_crypto_related or category == 'crypto':
                                news_item = {
                                    'title': entry.get('title', ''),
                                    'summary': entry.get('summary', ''),
                                    'link': entry.get('link', ''),
                                    'published': pub_date,
                                    'source': feed_url,
                                    'category': category
                                }
                                all_news.append(news_item)
                
                except Exception as e:
                    print(f"Error fetching from {feed_url}: {e}")
                    continue
        
        return pd.DataFrame(all_news)
    
    def analyze_sentiment(self, news_df):
        """Analyze sentiment of news articles"""
        if news_df.empty:
            return news_df
        
        sentiments = []
        
        for _, row in news_df.iterrows():
            # Combine title and summary for analysis
            text = f"{row.get('title', '')} {row.get('summary', '')}"
            
            if text.strip():
                try:
                    blob = TextBlob(text)
                    sentiment = blob.sentiment
                    sentiment_score = sentiment.polarity  # -1 (negative) to 1 (positive)
                    sentiment_label = self._classify_sentiment(sentiment_score)
                    
                    sentiments.append({
                        'sentiment_score': sentiment_score,
                        'sentiment_label': sentiment_label,
                        'subjectivity': sentiment.subjectivity
                    })
                except:
                    sentiments.append({
                        'sentiment_score': 0,
                        'sentiment_label': 'neutral',
                        'subjectivity': 0
                    })
            else:
                sentiments.append({
                    'sentiment_score': 0,
                    'sentiment_label': 'neutral',
                    'subjectivity': 0
                })
        
        # Add sentiment data to dataframe
        sentiment_df = pd.DataFrame(sentiments)
        result_df = pd.concat([news_df.reset_index(drop=True), sentiment_df], axis=1)
        
        return result_df
    
    def _classify_sentiment(self, score):
        """Classify sentiment score into categories"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def get_market_sentiment_score(self, news_df):
        """Calculate overall market sentiment score"""
        if news_df.empty:
            return {
                'overall_sentiment': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'confidence': 0
            }
        
        # Weight recent news more heavily
        now = datetime.now()
        news_df['hours_ago'] = news_df['published'].apply(
            lambda x: (now - x).total_seconds() / 3600
        )
        
        # Apply time-based weighting (more recent = higher weight)
        news_df['time_weight'] = np.exp(-news_df['hours_ago'] / 12)  # Decay over 12 hours
        
        # Calculate weighted sentiment
        weighted_sentiment = (news_df['sentiment_score'] * news_df['time_weight']).sum() / news_df['time_weight'].sum()
        
        # Count sentiment categories
        bullish_count = len(news_df[news_df['sentiment_label'] == 'positive'])
        bearish_count = len(news_df[news_df['sentiment_label'] == 'negative'])
        neutral_count = len(news_df[news_df['sentiment_label'] == 'neutral'])
        
        # Calculate confidence based on number of articles and subjectivity
        confidence = min(len(news_df) / 10, 1.0)  # Max confidence with 10+ articles
        avg_subjectivity = news_df['subjectivity'].mean()
        confidence *= (1 - avg_subjectivity)  # Lower confidence for subjective news
        
        return {
            'overall_sentiment': weighted_sentiment,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'confidence': confidence * 100
        }
    
    def get_high_impact_keywords(self, news_df):
        """Identify high-impact keywords that might affect trading"""
        if news_df.empty:
            return []
        
        high_impact_terms = [
            # Regulatory
            'regulation', 'ban', 'legal', 'sec', 'government', 'law',
            # Market events
            'etf', 'institutional', 'adoption', 'partnership', 'acquisition',
            # Technical events
            'halving', 'upgrade', 'fork', 'network', 'mining',
            # Market movements
            'breakout', 'resistance', 'support', 'rally', 'crash', 'surge'
        ]
        
        keyword_counts = {}
        for term in high_impact_terms:
            count = 0
            for _, row in news_df.iterrows():
                text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
                if term in text:
                    count += 1
            if count > 0:
                keyword_counts[term] = count
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:10]
    
    def create_news_features(self, news_df):
        """Create features for ML model based on news analysis"""
        sentiment_data = self.get_market_sentiment_score(news_df)
        
        features = {
            'news_sentiment_score': sentiment_data['overall_sentiment'],
            'news_bullish_ratio': sentiment_data['bullish_count'] / max(len(news_df), 1),
            'news_bearish_ratio': sentiment_data['bearish_count'] / max(len(news_df), 1),
            'news_confidence': sentiment_data['confidence'] / 100,
            'news_volume': len(news_df),  # Number of relevant news articles
        }
        
        # Recent news intensity (last 6 hours)
        if not news_df.empty:
            recent_cutoff = datetime.now() - timedelta(hours=6)
            recent_news = news_df[news_df['published'] >= recent_cutoff]
            features['recent_news_intensity'] = len(recent_news) / max(len(news_df), 1)
        else:
            features['recent_news_intensity'] = 0
        
        return features