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
from Classifier import *

"""
Version: 1.0
Last Updated: 16.2.2021
"""

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

def scrapeHebrewDYK(url, Classifier):
    output_text = ''
    knowledge_bites = []
    knowledge_bites_vectors = []
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, 'html.parser')
    all_lines = soup.find(id="bodyContent").find_all("p") # Find ps

    for li in all_lines:
        li_as_str = str(li)
        if li_as_str.startswith('<p>') and li_as_str.endswith('</p>'):
            li_without_li = remove_p(li_as_str)

            line_without_bolds_and_italics = remove_bolds_and_italics(li_without_li)
            line_without_anchor = remove_anchors(line_without_bolds_and_italics)
            line_without_span = remove_span(line_without_anchor)
            clean_line = remove_what_is_in_brackets(line_without_span)
            if len(clean_line.strip())>0:
                output_text += clean_line + '\n'
                #all_DYKs.append(clean_line)
                knowledge_bites.append(clean_line)
                knowledge_bites_vectors.append(Classifier.encode_text(clean_line))

    return knowledge_bites, knowledge_bites_vectors

def knowledgebites_to_text(knowledge_bites):
    out_text = ''
    for bite in knowledge_bites:
        out_text += bite + '\n'
    return out_text

def test_scrapeHebrewDYK():
    url = "https://he.wikipedia.org/wiki/ויקיפדיה:הידעת%3F"
    url = "https://he.wikipedia.org/wiki/ויקיפדיה:הידעת%3F/2015/ינואר"
    output_text = scrapeHebrewDYK(url)
    print(output_text)
    base_dir = os.getcwd()
    output_file_name = 'hebrew_DYK.txt'
    output_file_path = base_dir + '/' + output_file_name
    output_file_handle = codecs.open(output_file_path, 'w', "utf-8")
    output_file_handle.write(output_text)
    output_file_handle.close()

def test_scarpe_Wikquote():
    #url = 'https://en.wikiquote.org/wiki/Wikiquote:Quote_of_the_Day'
    #url = "https://he.wikiquote.org/wiki/אריך_קסטנר"
    hebrew_name = 'אריך קסטנר'
    output_text = scrapeHebrewWikipedia(hebrew_name)

    #text_to_write = knowledgebites_to_text(knowledge_bites)
    base_dir = os.getcwd()
    output_file_name = 'answers.txt'
    output_file_path = base_dir + '/' + output_file_name
    output_file_handle = codecs.open(output_file_path, 'w', "utf-8")
    output_file_handle.write(output_text)
    output_file_handle.close()
    print('ended!!')
    #scrape_Challanges_Bank()

def scrape_Batch_of_DYKs():

    knowledge_bites = []
    knowledge_bites_vectors = []


    base_dir = os.getcwd()

    # Load Classifier:
    classifier_pickle_file = 'classifier.pkl'
    base_path = os.getcwd()
    classifier_pickle_full_path = base_path + '/' + classifier_pickle_file
    pickle_file = open(classifier_pickle_full_path, 'rb')
    Classifier = pickle.load(pickle_file)
    pickle_file.close()

    years_list = ['2015', '2016', '2017', '2018', '2019', '2020']
    months_list =['ינואר','פברואר' , 'מרץ', 'אפריל', 'מאי', 'יוני','יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר','נובמבר', 'דצמבר']
    #years_list = ['2015', '2016']
    #months_list =['ינואר','פברואר']
    #years_list = ['2015']
    #months_list = ['ינואר', 'פברואר']

    for year in years_list:
        for month in months_list:
            #url = "https://he.wikipedia.org/wiki/ויקיפדיה:הידעת%3F/2015/ינואר"
            url = "https://he.wikipedia.org/wiki/ויקיפדיה:הידעת%3F/" + year + "/" + month
            print(url)
            current_knowledge_bites, current_knowledge_bites_vectors = scrapeHebrewDYK(url, Classifier)

            knowledge_bites.extend(current_knowledge_bites)
            knowledge_bites_vectors.extend(current_knowledge_bites_vectors)

    # Create Matix:
    knowledge_bites_matrix = np.array(knowledge_bites_vectors)

    # Save Results:
    # Knowledge bites:
    knowledge_bites_pickle_file = 'knowledge_bites.pkl'
    base_path = os.getcwd()
    knowledge_bites_pickle_full_path = base_path + '/' + knowledge_bites_pickle_file
    pickle_file = open(knowledge_bites_pickle_full_path, 'wb')
    pickle.dump(knowledge_bites, pickle_file)
    pickle_file.close()

    # Knowledge bites vectors:
    knowledge_bites_matrix_pickle_file = 'knowledge_bites_matrix.pkl'
    base_path = os.getcwd()
    knowledge_bites_matrix_pickle_full_path = base_path + '/' + knowledge_bites_matrix_pickle_file
    pickle_file = open(knowledge_bites_matrix_pickle_full_path, 'wb')
    pickle.dump(knowledge_bites_matrix, pickle_file)
    pickle_file.close()

    print('knowledge_bites')
    print(knowledge_bites)
    print('knowledge_bites_matrix')
    print(knowledge_bites_matrix)
    print(type(knowledge_bites_matrix))
    print(knowledge_bites_matrix.shape)
    print(type(knowledge_bites_vectors))
    print('succues!!')


if __name__ == '__main__':
    #test_scrapeHebrewDYK()
    scrape_Batch_of_DYKs()
