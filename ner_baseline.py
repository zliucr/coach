

from src.utils import init_experiment
from src.ner.baseline_loader import get_dataloader
from src.ner.baseline_model import ConceptTagger, BiLSTMCRFTagger
from src.ner.baseline_trainer import BaselineTrainer, BiLSTMCRFTrainer
from config import get_params

import numpy as np
from tqdm import tqdm

def run_baseline(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)

    # get dataloader
    dataloader_tr, dataloader_val, dataloader_test, vocab = get_dataloader(params.batch_size, bilstmcrf=params.bilstmcrf, n_samples=params.n_samples)

    # build model
    if params.bilstmcrf:
        ner_tagger = BiLSTMCRFTagger(params, vocab)
        ner_tagger.cuda()
        baseline_trainer = BiLSTMCRFTrainer(params, ner_tagger)
    else:
        concept_tagger = ConceptTagger(params, vocab)
        concept_tagger.cuda()
        baseline_trainer = BaselineTrainer(params, concept_tagger)
    
    for e in range(params.epoch):
        logger.info("============== epoch {} ==============".format(e+1))
        loss_list = []
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        for i, (X, lengths, y) in pbar:
            X, lengths = X.cuda(), lengths.cuda()

            loss = baseline_trainer.train_step(X, lengths, y)
            loss_list.append(loss)
            pbar.set_description("(Epoch {}) LOSS:{:.4f}".format((e+1), np.mean(loss_list)))
        
        logger.info("Finish training epoch {}. LOSS:{:.4f}".format((e+1), np.mean(loss_list)))

        logger.info("============== Evaluate Epoch {} ==============".format(e+1))
        f1_score, stop_training_flag = baseline_trainer.evaluate(dataloader_val, istestset=False)
        logger.info("Eval on dev set. Entity-F1: {:.4f}.".format(f1_score))

        f1_score, stop_training_flag = baseline_trainer.evaluate(dataloader_test, istestset=True)
        logger.info("Eval on test set. Entity-F1: {:.4f}.".format(f1_score))

        if stop_training_flag == True:
            break

if __name__ == "__main__":
    params = get_params()
    run_baseline(params)