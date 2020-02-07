# Stock-Market-Prediction
Stock Market Prediction using twitter Sentiment Analysis uses twitter data retrieved from the Twitter API to predict the sentiment of the tweets about specific companies. It uses the Support Vector Machine API for analysis the new tweets based on the past tweets experiences.



import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import csv
import pandas as pd

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

class TwitterClient(object):
	'''
	Generic Twitter Class for sentiment analysis.
	'''
	def __init__(self):
		'''
		Class constructor or initialization method.
		'''
		# keys and tokens from the Twitter Dev Console
		consumer_key = '7aQjGeA40zOrNbBJugWaiTgNi'
		consumer_secret = 'fYHkmC4mp3ppEJV1Xe88Gs7UzULC2MTTz4db17E1Ec9KUroqPw'
		access_token = '1195342982326538241-rRaZRQXZj0YgCmRADNz9V1LMBoBQoI'
		access_token_secret = 'hLHEOmKSM4K7yZLanHdKrD59HV4ozForHfNFInCw2NjNM'

		# attempt authentication
		try:
			self.auth = OAuthHandler(consumer_key, consumer_secret)
			# set access token and secret
			self.auth.set_access_token(access_token, access_token_secret)
			# create tweepy API object to fetch tweets
			self.api = tweepy.API(self.auth)
		except:
			print("Error: Authentication Failed")

#------------------

	def clean_tweets(self,tweet):
		# Happy Emoticons
		happy = set([
			':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
			':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
			'=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
			'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
			'<3'
		])

		# Sad Emoticons
		sad = set([
			':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
			':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
			':c', ':{', '>:\\', ';('
		])

		# Emoji patterns
		emoji_pattern = re.compile("["
								   u"\U0001F600-\U0001F64F"  # emoticons
								   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
								   u"\U0001F680-\U0001F6FF"  # transport & map symbols
								   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
								   u"\U00002702-\U000027B0"
								   u"\U000024C2-\U0001F251"
								   "]+", flags=re.UNICODE)

		# combine sad and happy emoticons
		emoticons = happy.union(sad)
		stop_words = set(stopwords.words('english'))
		word_tokens = word_tokenize(tweet)

		# after tweepy preprocessing the colon left remain after removing mentions
		# or RT sign in the beginning of the tweet
		tweet = re.sub(r':', '', tweet)
		tweet = re.sub(r'‚Ä¶', '', tweet)
		# replace consecutive non-ASCII characters with a space
		tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)

		# remove emojis from tweet
		tweet = emoji_pattern.sub(r'', tweet)

		# filter using NLTK library append it to a string
		filtered_tweet = [w for w in word_tokens if not w in stop_words]
		filtered_tweet = []

		# looping through conditions
		for w in word_tokens:
			# check tokens against stop words , emoticons and punctuations
			if w not in stop_words and w not in emoticons and w not in string.punctuation:
				filtered_tweet.append(w)
		return ' '.join(filtered_tweet)

	# print(word_tokens)
	# print(filtered_sentence)



#========
	def clean_tweet(self, tweet):
		'''
		Utility function to clean tweet text by removing links, special characters
		using simple regex statements.
		'''
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

	def get_tweet_sentiment(self, tweet):
		'''
		Utility function to classify sentiment of passed tweet
		using textblob's sentiment method
		'''
		# create TextBlob object of passed tweet text to predict the sentiment of the tweets 
    
		analysis = TextBlob(self.clean_tweet(tweet))
		# set sentiment
		if analysis.sentiment.polarity > 0:
			return 'positive'
		else:
			return 'negative'

	def get_tweets(self, query, count = 10):
		tweets = []

		try:
			#  twitter api to fetch tweets from the twitter
			fetched_tweets = self.api.search(q = query, count = count,lang = 'en')

			# parsing tweets one by one
			for tweet in fetched_tweets:
				# empty dictionary to store required params of a tweet
				parsed_tweet = {}

				# saving text of tweet
				parsed_tweet['text'] = tweet.text
				# saving sentiment of tweet
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

				# appending parsed tweet to tweets list
				if tweet.retweet_count > 0:
					# if tweet has retweets, ensure that it is appended only once
					if parsed_tweet not in tweets:
						tweets.append(parsed_tweet)
				else:
					tweets.append(parsed_tweet)

			# return parsed tweets
			return tweets

		except tweepy.TweepError as e:
			# print error (if any)
			print("Error : " + str(e))

def main():

    api = TwitterClient()
    tweets = api.get_tweets(query = 'Google', count = 1500)
    csvFile = open('result.csv', 'a')
    csvWriter = csv.writer(csvFile)

    for tweet in tweets:
        csvWriter.writerow([tweet['text'].encode('utf-8'),tweet['sentiment']])

    trainData = pd.read_csv("result.csv")
    vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
    train_vectors = vectorizer.fit_transform(trainData['text'])
    review = """Google Slides are  not that good"""

    classifier_linear = svm.SVC(kernel='linear',gamma=0.01,C=100,probability=True)
    classifier_linear.fit(train_vectors, trainData['sentiment'])

    review_vector = vectorizer.transform([review])
    print(classifier_linear.predict(review_vector))



if __name__ == "__main__":
	# calling main function
	main()
