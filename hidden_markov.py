import argparse
import numpy as np
from utils import make_dict, parse_file
import logging

                   
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


    logging.debug(f"Num Sentences: {len(sentences)}")
    logging.debug(f"Num Tags: {len(tags)}")

    np.savetxt(args.init, init)
    np.savetxt(args.emission, emission)
    np.savetxt(args.transition, transition)

    return
