# let the input doc be named input.txt
# Open the file and pre process it for RNN and save it as output.txt

import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

input_file = open('input.txt', 'r')
output_file = open('output.txt', 'w+')

input_lines = input_file.readlines()
tokenizer = RegexpTokenizer(r'\w+')
# read line by line and preprocess every line. (To avoid OOM issue and lag on reading large files)
for line in input_lines:
    # RegExpTokenizer removes punctuations also.
    tokens = tokenizer.tokenize(line)
    # removing stop words and numbers
    stop_words = stopwords.words('english')
    filtered_tokens = [t for t in tokens if t not in stop_words and not re.match('[0-9]', t)]
    # Lemmatization. (Stemming is ignored as i feel it's stupid)
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    # converting everything to lowercase
    filtered_tokens = [t.lower() for t in filtered_tokens]

    # write word by word to output.txt
    for word in filtered_tokens:
        output_file.write("%s\n" % word)

input_file.close()
output_file.flush()
output_file.close()
