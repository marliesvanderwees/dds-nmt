"""
Dynamic data selection for NMT
Author: Marlies van der Wees
For details, see paper 'Dynamic Data Selection for Neural Machine Translation'.
"""

# imports
import argparse
import sys, random, time
from numpy.random import choice


# functions
def parse_commandline():
    """
    Commandline argument parser
    """
    program = "Dynamic data selection for NMT as described in " +\
              "(van der Wees et al., EMNLP 2017)"
    parser  = argparse.ArgumentParser(prog=program)
    parser.add_argument("--bitext_src", required=True,
                        help="Path to source file of ranked bitext")
    parser.add_argument("--bitext_trg", required=True,
                        help="Path to target file of ranked bitext")
    parser.add_argument("--ced_weights", help="Path to file with CED weights of " +
                        "ranked bitext (only used for sampling)")
    parser.add_argument("--dds_method", required=True, choices=(["gft", "sampling"]),
                        help="Method to use for dynamic data selection")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Relative start size (range 0-1, default=0.5, " +
                             "used for gft and sampling)")
    parser.add_argument("--beta", type=float, default=0.7,
                        help="Retention rate (range 0-1, default=0.7, used for gft)")
    parser.add_argument("--eta", type=int, default=2,
                        help="Number of epochs to use each subset " + 
                             "(default=2, used for gft)")
    parser.add_argument("--sampling_fraction", type=float, default=0.2, 
                        help="Fraction of complete bitext to include in each sample " + 
                             "(range 0-1, default=0.2, used for sampling)")
    parser.add_argument("--total_epochs", type=int, default=16,
                        help="Total number of epochs to generate subsets for " +
                             "(default=16, used for gft and sampling)")
    return parser.parse_args()


def normalize_weights(weights):
    """
    Invert and min-max normalize CED weights as in Eq 3.
    """
    max_weight   = max(weights)
    min_weight   = min(weights)
    norm_weights = [1.0 - ((w - min_weight) / (max_weight - min_weight)) for w in weights]
    return norm_weights

    
def convert_weights_to_probabilities(weights):
    """
    Makes sure that the weights sum to 1 as in Eq 4.
    """
    sum_weights = sum(weights)
    return [float(i)/sum_weights for i in weights]


def sample_training_data(bitext_src, bitext_trg, weights_file, start_size, 
                         samp_fraction, total_epochs):
    """
    Apply sampling as described in Sec 3, Eq 3 and 4.
    """
    print("Sampling %0.1f%% of the training data for %d epochs" 
          %(100*samp_fraction, total_epochs))
          
    with open(bitext_src, "r") as src_in, open(bitext_trg, "r") as trg_in:
        src_lines = src_in.readlines()
        trg_lines = trg_in.readlines()

    with open(weights_file, "r") as weights:
        float_weights = [float(w) for w in weights]

    assert(len(src_lines) == len(trg_lines) and len(trg_lines) == len(float_weights))
    
    # absolute rather than relative hyperparam values
    select_from   = int(start_size * len(src_lines))
    num_to_select = int(samp_fraction * len(src_lines))
    
    # normalize weights
    top_n_src     = src_lines[:select_from]
    top_n_trg     = trg_lines[:select_from]
    top_n_weights = float_weights[:select_from]
    norm_weights  = normalize_weights(top_n_weights)
    sample_probs  = convert_weights_to_probabilities(norm_weights)

    # weighted sampling: draw n sentence pairs per epoch
    for n in range(1, total_epochs+1):
        epoch_nr = str(n)
        selection = choice(select_from, size=num_to_select, replace=False, p=sample_probs)

        # write selected sentences to train.src.epoch and train.trg.epoch
        with open(bitext_src + "." + epoch_nr, "w+") as src_out, \
             open(bitext_trg + "." + epoch_nr, "w+") as trg_out:
            for sent_nr in sorted(selection):
                src_out.write(src_lines[sent_nr])
                trg_out.write(trg_lines[sent_nr])

    
def gradual_fine_tuning(bitext_src, bitext_trg, start_size, retention_rate, 
                        num_epochs, total_epochs):
    """
    Apply gradual fine-tuning as described in Sec 3, Eq 5.
    """
    print("Applying gradual fine-tuning for %d epochs" %total_epochs)
    
    with open(bitext_src, "r") as src_in, open(bitext_trg, "r") as trg_in:
        src_lines = src_in.readlines()
        trg_lines = trg_in.readlines()
    
    assert(len(src_lines) == len(trg_lines))
    
    # selection fraction according to Eq 5
    fraction_to_keep = [int(len(src_lines) * start_size * retention_rate ** (i/num_epochs)) 
                        for i in range(total_epochs)]
    
    # write sentences to train.src.epoch and train.trg.epoch
    for n in range(total_epochs):
        epoch_nr = str(n + 1)
        top_n_to_keep = fraction_to_keep[n]
        with open(bitext_src + "." + epoch_nr, "w+") as src_out, \
             open(bitext_trg + "." + epoch_nr, "w+") as trg_out:
            for i in range(top_n_to_keep):
                src_out.write(src_lines[i])
                trg_out.write(trg_lines[i])        


# run program
def main():
    options = parse_commandline()
    bitext_src     = options.bitext_src
    bitext_trg     = options.bitext_trg
    bitext_weights = options.ced_weights
    start_size     = options.alpha
    retention_rate = options.beta
    num_epochs     = options.eta
    samp_fraction  = options.sampling_fraction 
    total_epochs   = options.total_epochs 
  
    # sanity checks
    if not (0.0 <= start_size and start_size <= 1.0):
        exit("Quitting program: Start size alpha should be in range 0.0-1.0")
    if not (0.0 <= retention_rate and retention_rate <= 1.0):
        exit("Quitting program: Retention rate beta should be in range 0.0-1.0")
    if not (0.0 <= samp_fraction and samp_fraction <= 1.0):
        exit("Quitting program: Sampling fraction should be in range 0.0-1.0")
 
    # dynamic data selection
    dds_method  = options.dds_method
    if dds_method == "gft":
        gradual_fine_tuning(bitext_src, bitext_trg, start_size, retention_rate, 
                            num_epochs, total_epochs)
    elif dds_method == "sampling":
        if not bitext_weights:
            exit("Quitting program: CED weights file not provided.")
        sample_training_data(bitext_src, bitext_trg, bitext_weights, start_size, 
                             samp_fraction, total_epochs)

if __name__ == "__main__":
  main()
