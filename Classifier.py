"""
This file was originally written by Netzer Bar Am
Version: 1.0
Last updated: 16.2.2021
"""

import requests
from bs4 import BeautifulSoup
import random
import re
import codecs
import os
import pickle
import os
import wikipedia
import codecs
import nltk
import string
import re
import pickle
import numpy as np
import sys
from requests.exceptions import ConnectionError


class Label(object):
    """This class holds and manages the attribute of a specific class"""
    def __init__(self, label_title, keywords, weight, not_allowed_tokens):
        self.label_title = label_title
        self.keywords = keywords
        self.weight = weight
        self.not_allowed_tokens = not_allowed_tokens

        self.word_freqs = {}
        self.vocab_of_label = []
        self.alpha = 1.  # Laplace smoothing
        self.prior_log_probability = 0.
        self.num_of_words_in_label = 0.
        self.num_of_word_types = 0
        self.total_vocab_size = 1.  # will be updated later

        self.word_log_probabilities = {}

        self.train()
        # print(self.word_freqs)
        print('# of token types:')
        print(len(self.word_freqs))

    def set_prior_log_probability(self, prior_log_probability):
        self.prior_log_probability = prior_log_probability

    def get_prior_log_probability(self):
        return self.prior_log_probability

    def is_good_token(self, token):
        token_is_allowed = token not in self.not_allowed_tokens
        token_is_not_a_number = not token.isnumeric()
        token_does_not_contain_a_numbers = self.token_does_not_contain_a_numbers(token)

        return token_is_allowed and token_is_not_a_number and token_does_not_contain_a_numbers

    def token_does_not_contain_a_numbers(self, token):
        for c in token:
            if c.isdigit():
                return False

        return True

    def train(self):
        for keyword in self.keywords:
            self.get_wiki_content(keyword)

        for word_type in self.word_freqs.keys():
            self.vocab_of_label.append(word_type)
            self.num_of_word_types += 1

    def get_word_freqs(self):
        return self.word_freqs.copy()

    def get_count_of_word(self, word):
        return self.word_freqs.get(word, 0)

    def get_wiki_content(self, keyword):
        try:

            url = hebrew_name_to_wikipedia_url(keyword)

            content = scrapeHebrewWikipedia(url)
            all_tokens = nltk.tokenize.word_tokenize(content)
            for token in all_tokens:
                lowered_token = token.lower()
                if self.is_good_token(lowered_token):
                    self.word_freqs[lowered_token] = self.word_freqs.get(lowered_token, 0)+1
                    self.num_of_words_in_label += 1
        except ConnectionError:  # This is the correct syntax
            print('*'*32)
            print('* Check Internet connection!!! *')
            print('*' * 32)
            sys.exit(1)

        except:
            print('error occured in ' + keyword)

    def set_word_log_probability(self, word):
        self.word_log_probabilities[word] = \
            np.log((self.alpha + self.get_count_of_word(word))/(self.alpha * self.total_vocab_size + self.num_of_words_in_label))

    def get_word_log_probability(self, word):
        return self.word_log_probabilities[word]

    def set_total_vocab_size(self, total_vocab_size):
        self.total_vocab_size = total_vocab_size

    def get_weight(self):
        return self.weight

    def get_label_title(self):
        return self.label_title

    def get_total_vocab_size(self):
        return self.total_vocab_size


class TextClassifier(object):
    """This class holds the atrributes of the Test Clasifier"""
    def __init__(self, input_labels_file_full_path, output_file_full_path, unallowed_words_file_full_path):
        self.input_labels_file_full_path = input_labels_file_full_path
        self.output_file_full_path = output_file_full_path
        self.unallowed_words_file_full_path = unallowed_words_file_full_path

        self.vocab = []
        self.total_vocab_size = 0
        self.not_allowed_tokens = []
        self.word_freqs = {}
        self.labels = []
        self.alpha_labels = 1  # Laplace smoothing
        self.total_sum_of_label_weights = 0

        self.collect_not_allowed_tokens()

        self.collect_labels()

        self.train()

    def train(self):
        for label in self.labels:
            label_word_freqs = label.get_word_freqs()
            for word, freq_of_word in label_word_freqs.items():
                self.word_freqs[word] = self.word_freqs.get(word, 0) + freq_of_word
                self.total_vocab_size += freq_of_word
                if word not in self.vocab:
                    self.vocab.append(word)

            prior_log_probability = np.log((self.alpha_labels + label.get_weight()) / (
                self.alpha_labels * len(self.labels) + self.total_sum_of_label_weights))
            label.set_prior_log_probability(prior_log_probability)

        for label in self.labels:
            label.set_total_vocab_size(self.total_vocab_size)
            for word in self.vocab:
                label.set_word_log_probability(word)

    def collect_labels(self):
        input_file_handle = codecs.open(self.input_labels_file_full_path, 'r', "utf-8")
        for line in input_file_handle.readlines():
            if len(line.strip())>0:
                fields = re.split(',|:', line.strip())
                # The first field is a name (non-predictive) and the last field is the label
                label_title, label_weight, label_keywords = fields[0], float(fields[1]), fields[2:]
                print('*'*30)
                print('label_title = %s' % label_title)
                print('label_weight = %s' % label_weight)
                print('label_keywords = %s' % label_keywords)

                self.total_sum_of_label_weights += label_weight
                list_of_keywords = []
                for keyword_option in label_keywords:
                    if len(keyword_option) > 1:
                        list_of_keywords.append(keyword_option.strip())

                print(list_of_keywords)

                self.labels.append(Label(label_title, list_of_keywords, label_weight, self.not_allowed_tokens))

        input_file_handle.close()

    def collect_not_allowed_tokens(self):
        input_file_handle = codecs.open(self.unallowed_words_file_full_path, 'r', "utf-8")
        for line in input_file_handle.readlines():
            list_of_words_in_line = line.split()
            for word in list_of_words_in_line:
                self.not_allowed_tokens.append(word)

        for t in string.punctuation:
            self.not_allowed_tokens.append(t)

        input_file_handle.close()

    def get_sentence_distribution(self, sentence):
        sentence_as_list = nltk.tokenize.word_tokenize(sentence)
        sentence_as_dict_of_frequencies = {}
        for token in sentence_as_list:
            if token not in self.not_allowed_tokens:
                word = token.lower()
                sentence_as_dict_of_frequencies[word] = sentence_as_dict_of_frequencies.get(word, 0) + 1

        labels_log_probabilities = {}
        for label in self.labels:

            log_probability = label.get_prior_log_probability()

            for word in self.vocab:
                word_occurances_in_sentence = sentence_as_dict_of_frequencies.get(word, 0)
                log_probability += label.get_word_log_probability(word) * word_occurances_in_sentence

            labels_log_probabilities[label.get_label_title()] = log_probability

        normalize_distribution = self.normalize_distribution(labels_log_probabilities)

        return normalize_distribution

    def normalize_distribution(self, labels_log_probabilities_non_normalized):
        #print('labels_log_probabilities_non_normalized')
        #print(labels_log_probabilities_non_normalized)

        max_log_p = max(labels_log_probabilities_non_normalized.values())
        #temp_d = {}
        #for label_title, log_p in labels_log_probabilities_non_normalized.items():
        #    temp_d[label_title] = log_p - max_log_p

        labels_log_probabilities_normalized = {}
        temp_sum = 0.
        for label_title, log_p in labels_log_probabilities_non_normalized.items():
        #for label_title, log_p in temp_d.items():
            temp_sum += np.exp(log_p - max_log_p)
            labels_log_probabilities_normalized[label_title] = log_p - max_log_p

        log_sum = np.log(temp_sum)
        for label_title, log_p in labels_log_probabilities_non_normalized.items():
        #for label_title, log_p in temp_d.items():
            labels_log_probabilities_normalized[label_title] -= log_sum


        #test_exp_sum(labels_log_probabilities_normalized)

        return labels_log_probabilities_normalized

    def classify(self, sentence):
        normalize_distribution = self.get_sentence_distribution(sentence)
        print('normalize_distribution')
        print(normalize_distribution)
        max_log_p = -1e100   # -inf
        for label, log_p in normalize_distribution.items():
            print(label, log_p)
            if max_log_p < log_p:
                predicted_class = label
                max_log_p = log_p

        print("The predicted class is: %s" % predicted_class)
        return predicted_class

    def encode_text(self, some_text):
        normalize_distribution = self.get_sentence_distribution(some_text)
        list_of_probabilities = []
        for label in self.labels:
            label_name = label.get_label_title()

            log_p = normalize_distribution[label_name]
            list_of_probabilities.append(np.exp(log_p))

        output_vec_of_probabilities = np.array(list_of_probabilities)
        return output_vec_of_probabilities

def test_exp_sum(dict_of_labels):
    temp_sum = 0.0
    for v in dict_of_labels.values():
        temp_sum += np.exp(v)

    print('should be 1:')
    print(temp_sum)

def remove_p(line):
    output_line = line[3:-4]
    return output_line


def remove_bolds_and_italics(line):
    l1 = line
    l2 = l1.replace('<b>','')
    l3 = l2.replace('</b>','')
    l4 = l3.replace('<i>','')
    l5 = l4.replace('</i>','')
    l6 = l5.replace('<br/>', '')
    l7 = l6.replace('<p>', '')
    l8 = l7.replace('</p>', '')
    return l8


def remove_what_is_in_brackets(input_line):
    output_line = input_line
    brackets_found = re.findall(r"\<.*\>", input_line)
    for part_to_remove in brackets_found:
        output_line = input_line.replace(part_to_remove, '')

    return output_line


def find_links(input_line):
    all_links_found = re.findall(r"href=\".*\" title", input_line)
    print(all_links_found)


def find_anchor_replacement(part_of_line):
    angle_bracket_positions = re.finditer('>', part_of_line)
    for chevron in angle_bracket_positions:
        start_of_cheveron_index, end_of_cheveron_index = chevron.span()

    return part_of_line[end_of_cheveron_index:]

def remove_anchors(line):
    output_line = ''

    starts_of_anchors_indicies = []
    ends_of_anchors_indicies = []
    starts_of_anchors = re.finditer('<a ', line)
    for i in starts_of_anchors:
        starts_of_anchors_indicies.append(i.span())
    end_of_anchors = re.finditer('</a>', line)
    for i in end_of_anchors:
        ends_of_anchors_indicies.append(i.span())

    current_index = 0
    e2 = 0
    for anchor_index in range(len(starts_of_anchors_indicies)):

        s1, e1 = starts_of_anchors_indicies[anchor_index]
        s2, e2 = ends_of_anchors_indicies[anchor_index]
        part_of_line = line[e1:s2]

        replacement = find_anchor_replacement(part_of_line)

        output_line += line[current_index:s1] + replacement
        current_index = e2

    output_line += line[e2:]

    return output_line


def remove_span(line):
    output_line = ''
    starts_of_spans_indicies = []
    ends_of_spans_indicies = []
    starts_of_spans = re.finditer('<span ', line)
    for i in starts_of_spans:
        starts_of_spans_indicies.append(i.span())
    end_of_spans = re.finditer('</span>', line)
    for i in end_of_spans:
        ends_of_spans_indicies.append(i.span())

    current_index = 0
    e2 = 0
    for span_index in range(len(starts_of_spans_indicies)):

        s1, e1 = starts_of_spans_indicies[span_index]
        s2, e2 = ends_of_spans_indicies[span_index]
        part_of_line = line[e1:s2]


        output_line += line[current_index:s1]
        current_index = e2

    output_line += line[e2:]

    return output_line


def train_classifier():

    force_retrain = True

    input_labels_file = 'labels.txt'
    unallowed_words_file = 'unallowed_words.txt'
    profile_file = 'profile.txt'
    output_file = 'output_file.txt'

    classifier_pickle_file = 'classifier.pkl'

    base_path = os.getcwd()
    classifier_pickle_full_path = base_path + '/' + classifier_pickle_file

    if os.path.isfile(classifier_pickle_full_path) and (not force_retrain):
        pickle_file = open(classifier_pickle_full_path, 'rb')
        Classifier = pickle.load(pickle_file)
        pickle_file.close()
    else:
        input_labels_file_full_path = base_path + '/' + input_labels_file
        output_file_full_path = base_path + '/' + output_file
        unallowed_words_file_full_path = base_path + '/' + unallowed_words_file
        Classifier = TextClassifier(input_labels_file_full_path, output_file_full_path, unallowed_words_file_full_path)

    sentence = 'אני מאד מתעניין בגיאוגרפיה. יש לי טלסקופ'
    print(sentence)
    Classifier.classify(sentence)
    Classifier.encode_text(sentence)

    sentence = 'אני אוהב לרוץ ולהתאמן בספורט'
    print(sentence)
    Classifier.classify(sentence)
    Classifier.encode_text(sentence)

    sentence = 'אני משוגע על פילוסופיה. ניטשה הוא הגיבור שלי!'
    print(sentence)
    Classifier.classify(sentence)
    Classifier.encode_text(sentence)

    sentence = 'אני קורא ספרים כפייתי. יש לי את כל כתבי שייקספיר על המדף.'
    print(sentence)
    Classifier.classify(sentence)
    Classifier.encode_text(sentence)

    ## Pickle (save) the result
    pickle_file = open(classifier_pickle_full_path, 'wb')
    pickle.dump(Classifier, pickle_file)
    pickle_file.close()


def scrapeHebrewWikipedia(url):
    output_text = ''
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, 'html.parser')
    all_lines = soup.find(id="bodyContent").find_all("p") # Find ps

    for li in all_lines:
        li_as_str = str(li)
        if li_as_str.startswith('<p>') and li_as_str.endswith('</p>'):
            li_without_li = remove_p(li_as_str)
            line_without_bolds_and_italics = remove_bolds_and_italics(li_without_li)
            #find_links(line_without_bolds_and_italics)
            line_without_anchor = remove_anchors(line_without_bolds_and_italics)
            line_without_span = remove_span(line_without_anchor)
            clean_line = remove_what_is_in_brackets(line_without_span)
            if len(clean_line.strip())>0:
                output_text += clean_line + '\n'

    return output_text

def hebrew_name_to_wikipedia_url(hebrew_name):
    hebrew_name_with_under_scores = hebrew_name.replace(' ','_')
    url = "https://he.wikipedia.org/wiki/" + hebrew_name_with_under_scores
    return url

def test_scrape_wikiquote():
    url = 'https://en.wikiquote.org/wiki/Wikiquote:Quote_of_the_Day'

    hebrew_name = 'אריך קסטנר'
    url = hebrew_name_to_wikipedia_url(hebrew_name)

    text_to_write = scrapeHebrewWikipedia(url)
    print(text_to_write)
    print(len(text_to_write))

    #text_to_write = knowledgebites_to_text(knowledge_bites)
    base_dir = os.getcwd()
    output_file_name = 'answers.txt'
    output_file_path = base_dir + '/' + output_file_name
    output_file_handle = codecs.open(output_file_path, 'w', "utf-8")
    output_file_handle.write(text_to_write)
    output_file_handle.close()
    print('ended!!')
    #scrape_Challanges_Bank()


if __name__ == '__main__':
    #test_scrape_wikiquote()
    train_classifier()