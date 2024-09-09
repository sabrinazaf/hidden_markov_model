
import numpy as np
from utils import make_dict, parse_file, get_matrices, write_predictions, write_metrics
import argparse
import logging

def mirrorbaby(betas,sentences,tags,word_dict,tag_dict):
    for i in reversed(range(len(sentences)-1)):
        w=word_dict[sentences[i+1]]
        for ii in range(0,len(tag_dict)):
            for iii in range(0,len(tag_dict)):
                betas[i,ii]=(transition[ii][iii]*betas[i+1,iii]*emission[iii,w])+betas[i,ii]
    return betas
def logtrick(arr):
    lmax=np.max(arr)
    count=0
    for i in arr:
        count+=np.exp(i-lmax)
    return lmax+np.log(count)


def main(args):
    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)
    # Parse the validation file
    sentences, tags = parse_file(args.validation_input)
    init, emission, transition = get_matrices(args)
    return sentences,tags,init,emission,transition,tag_dict,word_dict


def fwbw(sentences,tags,init,emission,transition,tag_dict,word_dict):
    linit=np.log(init)
    lemission=np.log(emission)
    ltransition=np.log(transition)
    betas=np.zeros((len(sentences),len(tag_dict)))
    for a in range(0, len(tag_dict)):
        betas[-1,a]=1
    alphas=np.zeros((len(sentences),len(tag_dict)))
    la=np.zeros((len(sentences),len(tag_dict)))
    predicted_tags=list()
    for i in range(len(tag_dict)):
        w=word_dict[sentences[0]]
        alphas[0,i]=emission[i,w]*init[i]
        la[0,i]=lemission[i,w]+linit[i]
    for aj in range(1,len(sentences)):
        w=word_dict[sentences[aj]]
        for ak in range(0,len(tag_dict)):
            alphsum=0
            lasum=np.zeros((len(tag_dict)))
            for al in range(0,len(tag_dict)):
                    alphsum=transition[al,ak]*alphas[aj-1,al]+alphsum
                    lasum[al]=ltransition[al,ak]+la[aj-1,al]
            alphas[aj,ak]=alphsum*emission[ak,w]
            la[aj,ak]=logtrick(lasum)+lemission[ak,w]
    betas=mirrorbaby(betas,sentences,tags,word_dict,tag_dict)
    logl=logtrick(la[-1])
    probs=np.multiply(alphas,betas)
    return logl,probs

def execute(transition,emission,init,sentences,tags, word_dict,tag_dict):
    loglikel=0
    revtag=dict((value, key) for key, value in tag_dict.items())
    predicted_tags=list()
    for i in range(0, len(sentences)):
        preds=list()
        stuff=fwbw(sentences[i],tags,init,emission,transition,tag_dict,word_dict)
        v=stuff[1]
        predy=np.argmax(stuff[1],axis=1)
        preds=[revtag.get(item,item)  for item in predy]
        predicted_tags.append(list(preds))
        loglikel+=stuff[0]

    avg_log_likelihood=loglikel/len(sentences)
    print(avg_log_likelihood)
    # Writing results to the corresponding files.
    accuracy = write_predictions(args.prediction_file, sentences, predicted_tags, tags)
    write_metrics(args.metric_file, avg_log_likelihood, accuracy)
   # print(avg_log_likelihood)
    accuracy = write_predictions(args.prediction_file, sentences, predicted_tags, tags)
    write_metrics(args.metric_file, avg_log_likelihood, accuracy)

    return
