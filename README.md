# Dynamic Data Selection for NMT

Code for applying dynamic data selection as described in [this paper](http://aclweb.org/anthology/D17-1148)
```
@inproceedings{wees2017dynamic,
  author    = {van der Wees, Marlies and Bisazza, Arianna and Monz, Christof},
  title     = {Dynamic Data Selection for Neural Machine Translation},
  booktitle = {Proceedings of EMNLP2017},
  year      = {2017},
  pages     = {1411--1421}
}
```

## Ranking a bitext by relevance to a given domain
Bilingual cross-entropy difference (CED)/Modified Moore-Lewis, as presented by 
Axelrod et al., Domain Adaptation via Pseudo In-Domain Data Selection, 2011

```
python scripts/rank-bitext.py --bitext_src=data/bitext.src --bitext_trg=data/bitext.trg --lm_domain_src=LM-I-SRC.blm --lm_domain_trg=LM-I-TRG.blm --lm_general_src=LM-G-SRC.blm --lm_general_trg=LM-G-TRG.blm
```

Produces:
- train.src and train.trg: ranked sentence pairs, most relevant sentences on top
- weights.txt: CED scores, one score per line, corresponding to sentence pairs in the bitext

## Dynamic data selection

Requires:
- ranked bitext files: train.src, train.trg, containing one sentence per line.
- specification of the DDS variant (sampling or gradual fine-tuning (gft))
- if applying sampling: a file with cross-entopy difference (CED) scores, one
   score per line corresponding to sentence pairs in the bitext.
   
Generates:
- training files for each epoch: train.src.1, train.trg.1, train.src.2, train.trg.2
 
**In order to use these training files, modify your favorite NMT code such that it reads in new training files for each epoch, while using source and target vocabulary of the complete set of training data used in all epochs.**

### Example call for gradual fine-tuning
```
$ python scripts/dynamic-data-selection.py --bitext_src=data/train.src --bitext_trg=data/train.trg --dds_method=gft --alpha=1 --beta=0.8 --eta=1 --total_epochs=12
```

Default parameter values that were used in the paper:
```
$ --alpha=0.5 --beta=0.7 --eta=2 --total_epochs=16
```

### Example call for sampling
```
$ python scripts/dynamic-data-selection.py --bitext_src=data/train.src --bitext_trg=data/train.trg --dds_method=sampling --ced_weights=data/weights.txt --alpha=1 --sampling_fraction=0.3 --total_epochs=12
```

Default parameter values that were used in the paper:
```
$ --alpha=0.5 --sampling_fraction=0.2 --total_epochs=16
```

