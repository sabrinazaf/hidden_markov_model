
import argparse
import numpy as np
from utils import make_dict, parse_file
import logging

# Setting up the argument parser
# don't change anything here

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to store the hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to store the hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to store the hmm_transition.txt (B) file')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')
def normalize(matrix):
    matlen=len(matrix)
    matrix=(matrix+1)
    matrix=matrix/np.sum(matrix, axis = 1).reshape(matlen, -1)
    return matrix


def main(args):
    sentences, tags = parse_file(args.train_input)
    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)
    init = np.zeros(len(tag_dict))
    emission =np.zeros((len(tag_dict),len(word_dict)))
    transition = np.zeros((len(tag_dict),len(tag_dict)))
    # Parse the train file
    # Suggestion: Take a minute to look at the training file,
    # it always hels to know your data :)
    sentences, tags = parse_file(args.train_input)
    count=0
    for i in range(len(sentences)):
        sentence=sentences[i]
        tagz=tags[i]
        for ii in range(len(sentence)):
            if (ii<1):
                init[tag_dict[tagz[ii]]]+=1
            emission[tag_dict[tagz[ii]]][word_dict[sentence[ii]]]+=1
            wordi=word_dict[sentence[ii]]
            tagi=tag_dict[tagz[ii]]
            if (len(sentence)-1>ii): #not reached end
                transition[tagi][tag_dict[tagz[ii+1]]]+=1
    init=(init+1)
    print(init)
    init=init/np.sum(init)
    emission=normalize(emission)
    transition=normalize(transition)
    print(transition[-1])
    print(transition)
    # Parse the train file
    # Suggestion: Take a minute to look at the training file,
    # it always hels to know your data :)


    logging.debug(f"Num Sentences: {len(sentences)}")
    logging.debug(f"Num Tags: {len(tags)}")

    np.savetxt(args.init, init)
    np.savetxt(args.emission, emission)
    np.savetxt(args.transition, transition)

    return

# No need to change anything beyond this point
if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)

 utils.py
 Download
import numpy as np

# This file is for your convenience.

def make_dict(file_path: str)->dict:
    '''
    Parses the file and returns a dictionary mapping of words/tags 
    to indices (line-number)
    ---
    Inputs:
        file_path: relative or absolute path of file to read

    Outputs:
        dictionary: Dictionary of the form {word: index}
    '''

    dictionary = None
    with open(file_path, "r") as f:
        lines = f.readlines()
        dictionary = {word.replace('\n', ''):k for k,word in enumerate(lines)}
    
    return dictionary

def parse_file(file_path: str)->tuple:
    '''
    Parses the file to return list of sentences and list of tags.
    ---
    Inputs:
        file_path: str, relative or absolute path of file to be parsed
    
    Outputs:
        sentences: list of lists
        tags: list of lists
    '''

    sentences = []
    tags = []
    
    with open(file_path, "r") as f:
        
        lines = f.readlines()
        
        sentence = []
        tag = []
        
        for pair in lines:
            
            if pair[0]=='\n':
                sentences.append(sentence)
                tags.append(tag)
                sentence = []
                tag = []

            else:
                word, y = pair.split('\t')
                y = y.replace('\n', '')
                sentence.append(word)
                tag.append(y)
                
        sentences.append(sentence)
        tags.append(tag)

    return sentences, tags

def write_predictions(prediction_file: str, sentences: list, predicted_tags: list, tags: list)->float:
    '''
    Function to write predictions to given file and return the accuracy
    ---
    Inputs:
        prediction_file: str, relative or absolute path of prediction file
        sentences: list of lists
        predicted_tags: list of lists
        tags: list of lists
    
    Outputs:
        sentences: list of lists
        tags: list of lists
    '''

    
    assert len(tags)==len(predicted_tags), f"Count mismatch. Check your predicted tags."
    assert type(sentences) == type(predicted_tags) == type(tags) == list, f"Type mistmatch"
    assert type(sentences[0]) == type(predicted_tags[0]) == type(tags[0]) == list, f"Type mistmatch"
    assert type(predicted_tags[0][0])== str, f"Predicted Tags should be string objects"
    
    accuracy = 0
    with open(prediction_file, 'w') as f:
        correct = 0
        total = 0
        for k,(sentence, tag_pred, tag) in enumerate(zip(sentences, predicted_tags, tags)):
            for word, y_hat, y in zip(sentence, tag_pred, tag):
                line = f"{word}\t{y_hat}\n"
                f.writelines(line)
                if y_hat == y:
                    correct += 1
                total += 1
            if k < len(tags)-1:
                f.writelines(f"\n")
        
        accuracy = correct / total
    
    return accuracy

def write_metrics(metric_file: str, avg_log_likelihood: float, accuracy: float)->None:
    '''
    Function to write predictions to given file and return the accuracy
    ---
    Inputs:
        metric_file: str, relative or absolute path of metrics file
        avg_log_likelihood: float
        accuracy: float

    Outputs:
        None
    '''
    
    if type(avg_log_likelihood) == np.float64:
        '''
        In case you left it as numpy's float64
        '''
        avg_log_likelihood = float(avg_log_likelihood)
    
    elif type(avg_log_likelihood) == np.float32:
        '''
        In case you left it as numpy's float32
        '''
        avg_log_likelihood = float(avg_log_likelihood)

    assert type(avg_log_likelihood) == float, "Log likelihood should be float"
    assert type(accuracy) == float, "Accuracy should be float"

    with open(metric_file, 'w') as f:

        likelihood = f"Average Log-Likelihood: {avg_log_likelihood:.16f}\n"
        acc = f"Accuracy: {accuracy:.16f}"
        f.writelines(likelihood)
        f.writelines(acc)

def get_matrices(args) -> tuple:
    '''
    Loads the three probability matrices.
    init, emit, and trans
    '''
    
    init = np.loadtxt(args.init)
    emission = np.loadtxt(args.emission)
    transition = np.loadtxt(args.transition)

    return init, emission, transition
