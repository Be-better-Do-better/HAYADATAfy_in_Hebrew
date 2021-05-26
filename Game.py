import os
import pickle
import numpy as np
import random
from bisect import bisect
from Classifier import *
import matplotlib.pyplot as plt

"""
Version: 2.2
Last Edited: 17.2.2021
"""

def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)


def weighted_choice(options, probabilities):

    total = 0
    cum_weights = []
    for p in probabilities:
        total += p
        cum_weights.append(total)
    x = random.random() * total
    i = bisect(cum_weights, x)

    selected_value = options[i]

    # max_index = np.where(probabilities == np.amax(probabilities))[0][0]
    # selected_value = options[max_index]

    return selected_value

def print_lines(line, window_length):
    end_of_window_length = round(3/4 * window_length)
    if len(line) <= window_length:
        print(line)
    else:
        space_index = line.rfind(' ',window_length-end_of_window_length, window_length)
        comma_index = line.rfind(',', window_length - end_of_window_length, window_length)
        period_index = line.rfind('. ', window_length - end_of_window_length, window_length)
        chosen_index = max([space_index, comma_index, period_index])

        if chosen_index == -1:
            chosen_index = window_length

        print(line[0:chosen_index])
        print_lines(line[chosen_index:], window_length)


class Game(object):
    """This class plays the game"""
    def __init__(self):
        self.state = 'init'
        self.profile = None
        self.current_knowledge_bite = ''

        self.liked_sentences = []

        self.Classifier = None
        self.knowledge_bites = None
        self.knowledge_bites_matrix = None
        self.user_weights = None
        self.matching_weights = None

        self.load_data()

        self.opening()
        self.create_profile()

        while self.state != 'quit':

            user_ans = self.get_user_input()

            if user_ans == 'q':
                self.state = 'quit'
                break

            elif user_ans == 'h':  # help = menu
                self.print_menu()

            elif user_ans == 'c':  # clear profile
                self.clear_profile()
                print('profile cleared')
                self.create_profile()

            elif user_ans == 'u': # update profile
                self.create_profile()

            elif user_ans == 'v':  # view profile
                self.view_profile()

            elif user_ans == '+': # add sentence to "liked":
                print('added to the "likes" :)')
                self.liked_sentences.append(self.current_knowledge_bite)

            elif user_ans == 'a':  # about
                self.about()

            elif user_ans == 'kkk':  # backdoor
                self.backdoor()

            else:
                self.fetch_knowledge()

        self.quit_game()

    def print_sign(self):
        print('                     ')
        print('     ++++++++++      ')
        print('   ++           ++   ')
        print('  ++              ++ ')
        print('                 ++  ')
        print('                ++   ')
        print('               ++    ')
        print('             ++      ')
        print('           ++        ')
        print('          ++         ')
        print('          ++         ')
        print('                     ')
        print('         +++         ')
        print('         +++         ')
        print('                     ')

    def opening(self):
        self.print_sign()
        #print('welcome to the HayaDATAfy!')
        print('ברוכים הבאים ל-הידעת-פי!')
        print('\n')

    def calc_user_weights(self):
        text_to_consider = self.profile + ' '
        for sentence in self.liked_sentences:
            text_to_consider += sentence + ' '

        self.user_weights = self.Classifier.encode_text(text_to_consider)

    def calc_bites_matching(self):
        narrow_interest_factor = 5.0 # 15 # 25.0 # 35.0
        self.matching_weights = softmax(narrow_interest_factor * np.matmul(self.knowledge_bites_matrix, self.user_weights.transpose()))

    def load_data(self):
        # Classifier:
        classifier_pickle_file = 'classifier.pkl'
        base_path = os.getcwd()
        classifier_pickle_full_path = base_path + '/' + classifier_pickle_file
        pickle_file = open(classifier_pickle_full_path, 'rb')
        self.Classifier = pickle.load(pickle_file)
        pickle_file.close()

        # Knowledge bites:
        knowledge_bites_pickle_file = 'knowledge_bites.pkl'
        knowledge_bites_pickle_full_path = base_path + '/' + knowledge_bites_pickle_file
        pickle_file = open(knowledge_bites_pickle_full_path, 'rb')
        self.knowledge_bites = pickle.load(pickle_file)
        pickle_file.close()

        # Knowledge bites matrix:
        knowledge_bites_matrix_pickle_file = 'knowledge_bites_matrix.pkl'
        knowledge_bites_matrix_pickle_full_path = base_path + '/' + knowledge_bites_matrix_pickle_file
        pickle_file = open(knowledge_bites_matrix_pickle_full_path, 'rb')
        self.knowledge_bites_matrix = pickle.load(pickle_file)
        pickle_file.close()

    def get_user_input(self):
        print('\n')
        #print("What do you want to do?")
        print("איך להמשיך מכאן?")
        #print("(q = exit, h = help, Enter = teach me!)")
        print("(q = exit, h = help, Enter = ספר לי עוד משהו)")
        #user_ans = input("(q = exit, h = help, Enter = teach me!)")
        user_ans = input("")
        #print("you entered: %s" % user_ans)
        return user_ans.lower()

    def print_menu(self):
        #print('Your options:')
        print('האפשרויות הן:')
        print('any-key + Enter = give me another knowledge bite')
        print('+ = like this piece!')
        print('c = clear profile')
        print('v = view your profile')
        print('u = update your profile')
        print('a = about')
        print('h = help')
        print('q = exit')

    def fetch_knowledge(self):
        print("הידעת...")
        window_length = 100
        self.current_knowledge_bite = weighted_choice(self.knowledge_bites, self.matching_weights)
        #print(self.current_knowledge_bite)
        print_lines(self.current_knowledge_bite, window_length)

    def clear_profile(self):
        self.profile = ''
        self.liked_sentences = []

    def quit_game(self):
        print('\n')
        self.print_sign()
        #print('Thanks and good-luck in the Hackathon...')
        print('מקוים שנהניתם,')
        print('\n'*2)
        #print('Bye Bye!')
        print('תודה ולהתראות!')

    def create_profile(self):
        #self.profile = input('Tell me about yourself: ')
        self.profile = input('ספר לי על עצמך: ')
        self.calc_user_weights()
        self.calc_bites_matching()

    def view_profile(self):
        print('\n')
        print("You wrote about yourself:")
        print(self.profile)
        text_to_consider = self.profile
        if len(self.liked_sentences) > 0:
            print('You liked the sentences:')
            for sentence in self.liked_sentences:
                print(sentence)
                text_to_consider += ' '+sentence

        normalize_distribution = self.Classifier.get_sentence_distribution(text_to_consider)
        list_of_labels = []
        list_of_interests = []
        for label, log_p in normalize_distribution.items():
            #list_of_labels.append(label)
            list_of_labels.append(label[::-1])
            list_of_interests.append(np.exp(log_p))

        plt.bar(list_of_labels, list_of_interests)
        plt.title('Profile Interests\n')
        plt.xticks(rotation=45)
        #plt.ylabel('probability')
        plt.ylabel('הסתברות'[::-1])
        plt.show()

    def about(self):
        all_about = "This program was written for the Albert Einstein Hackathon..."
        print(all_about)
        print("The # of knowledge bites is: %s" % len(self.knowledge_bites))

    def backdoor(self):
        self.Classifier.classify(self.profile)

def run_Game():
    my_game = Game()

if __name__ == '__main__':
    run_Game()