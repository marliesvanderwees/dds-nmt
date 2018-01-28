import argparse

def parse_commandline():
    """
    Commandline argument parser
    """
    program = "Bitext ranking for MT as described in " +\
              "(van der Wees et al., EMNLP 2017), " +\
              "following (Axelrod et al., EMNLP 2011)"
    parser  = argparse.ArgumentParser(prog=program)
    parser.add_argument("--bitext_files",
                        help="Comma-separated list of bitext files to rank")
    parser.add_argument("--src_domain_loss", required=True,
                        help="Path to sentence-level cross-entropy scores with " +\
                             "in-domain source LM")
    parser.add_argument("--trg_domain_loss", required=True,
                        help="Path to sentence-level cross-entropy scores with " +\
                             "in-domain target LM")
    parser.add_argument("--src_general_loss", required=True,
                        help="Path to sentence-level cross-entropy scores with " +\
                             "general-domain source LM")
    parser.add_argument("--trg_general_loss", required=True,
                        help="Path to sentence-level cross-entropy scores with " +\
                             "general-domain target LM")
    return parser.parse_args()

def compute_bilingual_ced_diff(src_domain_loss, trg_domain_loss, src_general_loss, trg_general_loss):
    """
    Computes difference between in-domain and general-domain loss score.
    Returns list of tuples (score, sent_id).
    """
    loss_scores = []
    with open(src_domain_loss, "r") as src_domain, open(trg_domain_loss, "r") as trg_domain, \
         open(src_general_loss, "r") as src_general, open(trg_general_loss, "r") as trg_general:
        
        sent_id = 0
        for sd, td, sg, tg in zip(src_domain, trg_domain, src_general, trg_general):
            combined_score = (float(sd) - float(sg)) + (float(td) - float(tg))
            loss_scores.append((combined_score, sent_id))
            sent_id += 1

    return (sorted(loss_scores))

def rank_sentences(sorted_ced_diff_scores, bitext_files):
    """
    Ranks sentences in bitext files according to sorted CED difference scores.
    Writes sorted sentences to output files.
    """
    sorted_sentence_numbers = [i[1] for i in sorted_ced_diff_scores]
    
    # Compute 1 - min-max normalized score as weight and save weight per line to weights file
    min_score = sorted_ced_diff_scores[0][0]
    max_score = sorted_ced_diff_scores[-1][0]
    normalized_ced_diff_scores = [1.0-(i[0]-min_score)/(max_score-min_score) for i in sorted_ced_diff_scores]
    
    with open("ranked-bitext.weights" , "w+") as weights_file:
        for weight in normalized_ced_diff_scores:
            weights_file.write("%0.3f\n" %weight)

    # Write selected sentences in weighted order to output file
    for bitext_file in bitext_files:
        with open(bitext_file) as train_data:
            train_sentences = train_data.readlines()
            if not len(train_sentences) == len(sorted_ced_diff_scores):
                exit("Exiting...  Bitext file %s does not have the expected number of lines " %bitext_file +\
                     "(%d instead of %d)" %(len(train_sentences), len(sorted_ced_diff_scores)))
        
        outfile_name = bitext_file + ".ranked"
        with open(outfile_name, "w+") as outfile:
            for sent_nr in sorted_sentence_numbers:
                outfile.write(train_sentences[sent_nr])


if __name__ == "__main__":
    options = parse_commandline()
    src_domain_loss  = options.src_domain_loss
    trg_domain_loss  = options.trg_domain_loss
    src_general_loss = options.src_general_loss
    trg_general_loss = options.trg_general_loss
    bitext_files     = [f for f in options.bitext_files.split(",")]
    
    if len(bitext_files) < 2:
        exit("Exiting... Please specify at least two bitext files to rank")
   
    sorted_ced_diff_scores = compute_bilingual_ced_diff(src_domain_loss, 
                                                        trg_domain_loss, 
                                                        src_general_loss, 
                                                        trg_general_loss)

    rank_sentences(sorted_ced_diff_scores, bitext_files)
