import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sb
from collections import Counter
import tqdm
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from wordcloud import WordCloud, STOPWORDS
import gensim
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_recall_fscore_support
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score