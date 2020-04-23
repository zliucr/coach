
from src.utils import init_experiment
from src.ner.dataloader import get_dataloader
from src.ner.trainer import NERTrainer
from src.ner.model import BinaryNERagger, EntityNamePredictor, SentRepreGenerator
from config import get_params

import numpy as np
from tqdm import tqdm

def main(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    
    # get dataloader
    dataloader_tr, dataloader_val, dataloader_test, vocab = get_dataloader(params.batch_size, use_label_encoder=params.tr, n_samples=params.n_samples)

    # build model
    binary_nertagger = BinaryNERagger(params, vocab)
    entityname_predictor = EntityNamePredictor(params)
    binary_nertagger, entityname_predictor = binary_nertagger.cuda(), entityname_predictor.cuda()

    if params.tr:
        sent_repre_generator = SentRepreGenerator(params, vocab)
        sent_repre_generator = sent_repre_generator.cuda()
        ner_trainer = NERTrainer(params, binary_nertagger, entityname_predictor, sent_repre_generator)
    else:
        ner_trainer = NERTrainer(params, binary_nertagger, entityname_predictor)
    
    for e in range(params.epoch):
        logger.info("============== epoch {} ==============".format(e+1))
        loss_bin_list, loss_entityname_list = [], []
        if params.tr:
            loss_tem0_list, loss_tem1_list = [], []
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        if params.tr:
            for i, (X, lengths, y_bin, y_final, templates, tem_lengths) in pbar:
                X, lengths, templates, tem_lengths = X.cuda(), lengths.cuda(), templates.cuda(), tem_lengths.cuda()
                loss_bin, loss_entityname, loss_tem0, loss_tem1 = ner_trainer.train_step(X, lengths, y_bin, y_final, templates=templates, tem_lengths=tem_lengths, epoch=e)
                loss_bin_list.append(loss_bin)
                loss_entityname_list.append(loss_entityname)
                loss_tem0_list.append(loss_tem0)
                loss_tem1_list.append(loss_tem1)

                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f} LOSS entity:{:.4f} LOSS TEM0:{:.4f} LOSS TEM1:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_entityname_list), np.mean(loss_tem0_list), np.mean(loss_tem1_list)))
        else:
            for i, (X, lengths, y_bin, y_final) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss_bin, loss_entityname = ner_trainer.train_step(X, lengths, y_bin, y_final)
                loss_bin_list.append(loss_bin)
                loss_entityname_list.append(loss_entityname)
                
                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f} LOSS entity:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_entityname_list)))
            
        logger.info("Finish training epoch {}. LOSS BIN:{:.4f} LOSS entity:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_entityname_list)))

        logger.info("============== Evaluate Epoch {} ==============".format(e+1))
        bin_f1, final_f1, stop_training_flag = ner_trainer.evaluate(dataloader_val, istestset=False)
        logger.info("Eval on dev set. Binary entity-F1: {:.4f}. Final entity-F1: {:.4f}.".format(bin_f1, final_f1))

        bin_f1, final_f1, stop_training_flag = ner_trainer.evaluate(dataloader_test, istestset=True)
        logger.info("Eval on test set. Binary entity-F1: {:.4f}. Final entity-F1: {:.4f}.".format(bin_f1, final_f1))

        if stop_training_flag == True:
            break

if __name__ == "__main__":
    params = get_params()
    main(params)
