import numpy as np
import pandas as pd
import re
from unidecode import unidecode

# Replace all numeric with 'n'
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

# redundancy words and their right formats
redundancy_rightFormat = {
	'ckckck': 'cock',
	'fuckfuck': 'fuck',
	'lolol': 'lol',
	'lollol': 'lol',
	'pussyfuck': 'fuck',
	'gaygay': 'gay',
	'haha': 'ha',
	'sucksuck': 'suck'}

redundancy = set(redundancy_rightFormat.keys())

# all the words below are included in glove dictionary
# combine these toxic indicators with 'CommProcess.revise_triple_and_more_letters'
toxic_indicator_words = [
	'fuck', 'fucking', 'fucked', 'fuckin', 'fucka', 'fucker', 'fucks', 'fuckers',
	'fck', 'fcking', 'fcked', 'fckin', 'fcker', 'fcks',
	'fuk', 'fuking', 'fuked', 'fukin', 'fuker', 'fuks', 'fukers',
	'fk', 'fking', 'fked', 'fkin', 'fker', 'fks',
	'shit', 'shitty', 'shite',
	'stupid', 'stupids',
	'idiot', 'idiots',
	'suck', 'sucker', 'sucks', 'sucka', 'sucked', 'sucking',
	'ass', 'asses', 'asshole', 'assholes', 'ashole', 'asholes',
	'gay', 'gays',
	'niga', 'nigga', 'nigar', 'niggar', 'niger', 'nigger',
	'monster', 'monsters',
	'loser', 'losers',
	'nazi', 'nazis',
	'cock', 'cocks', 'cocker', 'cockers',
	'faggot', 'faggy',
]
toxic_indicator_words_sets = set(toxic_indicator_words)

# Many toxic words in train and test data set are in various format (e.g. 'fuuuuck', 'fuckkkkk') and could not be
# found in pre-trained word vector.
# There is a ground truth that few english word contains more than three consecutive identical letter.
# Revise all consecutive identical letter more than three times to only two.
# For example, 'fuuuuck' → 'fuuck' and 'fuckkkkk' → ‘fuckk’


def _get_toxicIndicator_transformers():
	toxicIndicator_transformers = dict()
	for word in toxic_indicator_words:
		tmp_1 = []
		for char in word:
			if len(tmp_1) > 0:
				tmp_2 = []
				for pre in tmp_1:
					tmp_2.append(pre + char)
					tmp_2.append(pre + char + char)
				tmp_1 = tmp_2
			else:
				tmp_1.append(char)
				tmp_1.append(char + char)
		toxicIndicator_transformers[word] = tmp_1
	return toxicIndicator_transformers


# Dict data structure maps 'fuuck' to 'fuck' and 'fuckk to 'fuck'. In this way, 'fuuck' or 'fuckk' could be embedded
# via same word vector as 'fuck' when building model.
toxicIndicator_transformers = _get_toxicIndicator_transformers()

deny_origin = {
	"you're": ['you', 'are'],
	"i'm": ['i', 'am'],
	"he's": ['he', 'is'],
	"she's": ['she', 'is'],
	"it's": ['it', 'is'],
	"they're": ['they', 'are'],
	"can't": ['can', 'not'],
	"couldn't": ['could', 'not'],
	"don't": ['do', 'not'],
	"don;t": ['do', 'not'],
	"didn't": ['did', 'not'],
	"doesn't": ['does', 'not'],
	"isn't": ['is', 'not'],
	"wasn't": ['was', 'not'],
	"aren't": ['are', 'not'],
	"weren't": ['were', 'not'],
	"won't": ['will', 'not'],
	"wouldn't": ['would', 'not'],
	"hasn't": ['has', 'not'],
	"haven't": ['have', 'not'],
	"what's": ['what', 'is'],
	"that's": ['that', 'is'],
}
denies = set(deny_origin.keys())


class PreProcessComments(object):
	@staticmethod
	def clean_text(t):
		# replace any character other than [A-Za-z0-9,!?*.;’´'\/] with a space
		t = re.sub(r"[^A-Za-z0-9,!?*.;’´'\/]", " ", t)
		# replace all numerics with a space
		t = replace_numbers.sub(" ", t)
		# convert to lower case
		t = t.lower()
		t = re.sub(r",", " ", t)
		t = re.sub(r"’", "'", t)
		t = re.sub(r"´", "'", t)
		t = re.sub(r"\.", " ", t)
		t = re.sub(r"!", " ! ", t)
		t = re.sub(r"\?", " ? ", t)
		t = re.sub(r"\/", " ", t)
		return t

	@staticmethod
	def revise_deny(t):
		# replace words such as you're and i'm with its expanded forms -'you are' and 'i am'
		ret = []
		for word in t.split():
			if word in denies:
				ret.append(deny_origin[word][0])
				ret.append(deny_origin[word][1])
			else:
				ret.append(word)
		ret = ' '.join(ret)
		ret = re.sub("'", " ", ret)
		ret = re.sub(r";", " ", ret)
		return ret

	@staticmethod
	def revise_star(t):
		ret = []
		for word in t.split():
			# if the word contains a * remove all the stars and check whether the word exists in the
			# toxic_indicator_words_sets
			if ('*' in word) and (re.sub('\*', '', word) in toxic_indicator_words_sets):
				# remove all the starts
				word = re.sub('\*', '', word)
			ret.append(word)
		# if there are words with star, replace those starts with a space
		ret = re.sub('\*', ' ', ' '.join(ret))
		return ret

	@staticmethod
	# when a character repeats more than twice in a word, restrict it to 2.
	# for instance if there is a word 'whhhhyyy' it will be converted to 'whhyy'
	def revise_triple_and_more_letters(t):
		for letter in 'abcdefghijklmnopqrstuvwxyz':
			reg = letter + "{2,}"
			t = re.sub(reg, letter + letter, t)
		return t

	@staticmethod
	# replace words such as haha with ha
	def revise_redundancy_words(t):
		ret = []
		for word in t.split(' '):
			for redu in redundancy:
				if redu in word:
					word = redundancy_rightFormat[redu]
					break
			ret.append(word)
		return ' '.join(ret)

	@staticmethod
	def fill_na(t):
		if t.strip() == '':
			return 'NA'
		return t

	@staticmethod
	# convert text data in Unicode to its nearest representation in ASCII
	def convert_unicode_to_ascii(t):
		return unidecode(t)


def load_embeddings():
	f = open(F_EMBEDDING_FILE, 'r', encoding='utf-8')
	for line in f:
		values = line.split()
		try:
			fasttext_index.add(values[0])
		except:
			print("Err on ", values[:3])
	f.close()

	# f = open(G_EMBEDDING_FILE, 'r', encoding='utf-8')
	# for line in f:
	#     values = line.split()
	#     try:
	#         glove_index.add(values[0])
	#     except:
	#         print("Err on ", values[:3])
	# f.close()


def count_unknown_glove(t):
	t = t.split()
	res = 0
	for w in t:
		if w not in glove_index:
			res += 1
	return res


def count_unknown_fasttext(t):
	t = t.split()
	res = 0
	for w in t:
		if w not in fasttext_index:
			res += 1
	return res


def count_regexp_occ(regexp="", text=None):
	""" Simple way to get the number of occurence of a regex"""
	if len(text) == 0:
		return 0
	else:
		return len(re.findall(regexp, text)) / len(text)


def add_features(df):
	# angry people write short messages
	df['total_length'] = df[COMMENT_COL].apply(len)
	# many toxic comments were ALL CAPS
	df['capitals'] = df[COMMENT_COL].apply(
	    lambda comment: sum(1 for c in comment if c.isupper()))
	# many toxic comments were ALL CAPS
	df['caps_vs_length'] = df.apply(lambda row: float(
	    row['capitals'])/float(row['total_length']), axis=1)
	# several toxic comments had multiple exclamation marks
	df['num_exclamation_marks'] = df[COMMENT_COL].apply(
	    lambda comment: comment.count('!'))
	# angry people might not use question marks
	df['num_question_marks'] = df[COMMENT_COL].apply(
	    lambda comment: comment.count('?'))
	# angry people might not use punctuation
	df['num_punctuation'] = df[COMMENT_COL].apply(
		lambda comment: sum(comment.count(w) for w in '.,;:'))
	# words like fck or $# or sh*t mean more symbols in foul language
	df['num_symbols'] = df[COMMENT_COL].apply(
		lambda comment: sum(comment.count(w) for w in '*&$%'))
	# angry people might write short messages?
	df['num_words'] = df[COMMENT_COL].apply(lambda comment: len(comment.split()))
	# angry comments are sometimes repeated many times
	df['num_unique_words'] = df[COMMENT_COL].apply(
		lambda comment: len(set(w for w in comment.split())))
	# angry comments are sometimes repeated many times
	df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
	# Angry people wouldn't use happy smilies
	df['num_smilies'] = df[COMMENT_COL].apply(
		lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
	df['ant_slash_n'] = df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"\n", x))
	# Check number of upper case, if you're angry you may write in upper case
	# Number of F words - f..k contains folk, fork,
	df["nb_fk"] = df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
	# Number of S word
	df["nb_sk"] = df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
	# Number of D words
	df["nb_dk"] = df[COMMENT_COL].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
	# Number of occurrence of You, insulting someone usually needs someone called : you
	df["nb_you"] = df[COMMENT_COL].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
	# Just checking for toxic 19th century vocabulary
	df["nb_ng"] = df[COMMENT_COL].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
	# Just to check you really referred to mother
	df["nb_mother"]=df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"\Wmother\W", x))
	# Some Sentences start with a <:> so it may help
	df["start_with_columns"]=df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"^\:+", x))
	# Check for time stamp
	df["has_timestamp"]=df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
	# Check for dates 18:44, 8 December 2010
	df["has_date_long"]=df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
	# Check for date short 8 December 2010
	df["has_date_short"]=df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
	# Check for http links
	df["has_http"]=df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
	# check for mail
	df["has_mail"]=df[COMMENT_COL].apply(
		lambda x: count_regexp_occ(
		    r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
	)
	df["has_image"]=df[COMMENT_COL].apply(
		lambda x: count_regexp_occ(r'image\:', x)
	)
	df["has_ip"]=df[COMMENT_COL].apply(lambda x: count_regexp_occ(
	    "(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", x))
	# Looking for words surrounded by == word == or """" word """"
	df["has_emphasize_equal"]=df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
	df["has_emphasize_quotes"]=df[COMMENT_COL].apply(
	    lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))
	df["has_star"]=df[COMMENT_COL].apply(lambda x: count_regexp_occ(r"\*", x))
	# df["unknown_glove"] = df[CLEAN_COMMENT_COL].apply(lambda x: count_unknown_glove(x))
	df["unknown_fasttext"]=df[CLEAN_COMMENT_COL].apply(
	    lambda x: count_unknown_fasttext(x))
	# df["unknown_glove_fasttext"] = df["unknown_glove"] + df["unknown_fasttext"]
	return df

COMMENT_COL='comment_text'
CLEAN_COMMENT_COL='clean_text'

glove_index = set()
fasttext_index = set()
F_EMBEDDING_FILE = 'Data/embeddings/crawl-300d-2M.vec'
G_EMBEDDING_FILE = 'Data/embeddings/glove.840B.300d.txt'

def execute_pre_process(df):
	# make a copy of the comment_text. We will keep the comment_text untouched and use
	# clean_text for preprocessing
	df[CLEAN_COMMENT_COL]=df[COMMENT_COL]
	
	load_embeddings()
	
	pre_process_pipeline=[
		# convert Unicode to its nearest representation in ASCII
		PreProcessComments.convert_unicode_to_ascii,
		# replace certain characters with space
		PreProcessComments.clean_text,
		# replace words such as you're and i'm with its expanded forms -'you are' and 'i am'
		PreProcessComments.revise_deny,
		# remove stars if the word formed after removing the start is part of toxic_indicator_words_sets
		# otherwise replace stars with spaces
		PreProcessComments.revise_star,
		# when a character repeats more than twice in a word, restrict it to 2.
		PreProcessComments.revise_triple_and_more_letters,
		# replace words such as haha with ha
		PreProcessComments.revise_redundancy_words,
		# replace missing comments with NA
		PreProcessComments.fill_na,
	]
	for pre_process_func in pre_process_pipeline:
		df[CLEAN_COMMENT_COL]=df[CLEAN_COMMENT_COL].apply(pre_process_func)

	df = add_features(df)
	return df

# # Process whole train data
# print('Pre Processing train data')
# input_dir = 'Data/'
# df_train = pd.read_csv(input_dir + 'train.csv')
# df_train = execute_pre_process(df_train)
# print(df_train.shape)
# df_train.to_csv(input_dir + 'train_processed.csv', index=False)
