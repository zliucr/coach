
from src.utils import init_experiment
from src.slu.datareader import datareader, read_file, binarize_data
from src.slu.dataloader import get_dataloader, Dataset, DataLoader, collate_fn
from src.slu.baseline_loader import get_dataloader as get_baselineloader
from src.slu.baseline_loader import collate_fn as baseline_collate_fn
from src.slu.baseline_loader import Dataset as BaselineDataset
from src.slu.trainer import SLUTrainer
from src.slu.baseline_trainer import BaselineTrainer
from config import get_params
import torch
import os


def test_coach(params):
    # get dataloader
    _, _, dataloader_test, _ = get_dataloader(params.tgt_dm, params.batch_size, params.tr, params.n_samples)

    model_path = params.model_path
    assert os.path.isfile(model_path)
    
    reloaded = torch.load(model_path)
    binary_slu_tagger = reloaded["binary_slu_tagger"]
    slotname_predictor = reloaded["slotname_predictor"]
    binary_slu_tagger.cuda()
    slotname_predictor.cuda()

    slu_trainer = SLUTrainer(params, binary_slu_tagger, slotname_predictor)

    _, f1_score, _ = slu_trainer.evaluate(dataloader_test, istestset=True)
    print("Eval on test set. Final Slot F1 Score: {:.4f}.".format(f1_score))


def test_baseline(params):
    # get dataloader
    _, _, dataloader_test, _ = get_baselineloader(params.tgt_dm, params.batch_size, params.n_samples)

    model_path = params.model_path
    assert os.path.isfile(model_path)

    reloaded = torch.load(model_path)
    slu_tagger = reloaded["slu_tagger"]
    slu_tagger.cuda()

    baseline_trainer = BaselineTrainer(params, slu_tagger)

    _, f1_score, _ = baseline_trainer.evaluate(0, dataloader_test, istestset=True)
    print("Eval on test set. Slot F1 Score: {:.4f}.".format(f1_score))


def test_coach_on_seen_and_unseen(params):
    # read seen and unseen data
    print("Getting vocabulary ...")
    _, vocab = datareader(params.tr)

    print("Processing Unseen and Seen samples in %s domain ..." % params.tgt_dm)
    unseen_data, vocab = read_file("data/snips/"+params.tgt_dm+"/unseen_slots.txt", vocab, False)
    seen_data, vocab = read_file("data/snips/"+params.tgt_dm+"/seen_slots.txt", vocab, False)

    print("Binarizing data ...")
    if len(unseen_data["utter"]) > 0:
        unseen_data_bin = binarize_data(unseen_data, vocab, params.tgt_dm, False)
    else:
        unseen_data_bin = None
    
    if len(seen_data["utter"]) > 0:
        seen_data_bin = binarize_data(seen_data, vocab, params.tgt_dm, False)
    else:
        seen_data_bin = None

    model_path = params.model_path
    assert os.path.isfile(model_path)
    reloaded = torch.load(model_path)
    binary_slu_tagger = reloaded["binary_slu_tagger"]
    slotname_predictor = reloaded["slotname_predictor"]
    binary_slu_tagger.cuda()
    slotname_predictor.cuda()

    slu_trainer = SLUTrainer(params, binary_slu_tagger, slotname_predictor)

    print("Prepare dataloader ...")
    if unseen_data_bin:
        unseen_dataset = Dataset(unseen_data_bin["utter"], unseen_data_bin["y1"], unseen_data_bin["y2"], unseen_data_bin["domains"])

        unseen_dataloader = DataLoader(dataset=unseen_dataset, batch_size=params.batch_size, collate_fn=collate_fn, shuffle=False)

        _, f1_score, _ = slu_trainer.evaluate(unseen_dataloader, istestset=True)
        print("Evaluate on {} domain unseen slots. Final slot F1 score: {:.4f}.".format(params.tgt_dm, f1_score))

    else:
        print("Number of unseen sample is zero")

    if seen_data_bin:
        seen_dataset = Dataset(seen_data_bin["utter"], seen_data_bin["y1"], seen_data_bin["y2"], seen_data_bin["domains"])
        
        seen_dataloader = DataLoader(dataset=seen_dataset, batch_size=params.batch_size, collate_fn=collate_fn, shuffle=False)

        _, f1_score, _ = slu_trainer.evaluate(seen_dataloader, istestset=True)
        print("Evaluate on {} domain seen slots. Final slot F1 score: {:.4f}.".format(params.tgt_dm, f1_score))

    else:
        print("Number of seen sample is zero")


def test_baseline_on_seen_and_unseen(params):
    # read seen and unseen data
    print("Getting vocabulary ...")
    _, vocab = datareader()

    print("Processing Unseen and Seen samples in %s domain ..." % params.tgt_dm)
    unseen_data, vocab = read_file("data/snips/"+params.tgt_dm+"/unseen_slots.txt", vocab, False)
    seen_data, vocab = read_file("data/snips/"+params.tgt_dm+"/seen_slots.txt", vocab, False)

    print("Binarizing data ...")
    if len(unseen_data["utter"]) > 0:
        unseen_data_bin = binarize_data(unseen_data, vocab, params.tgt_dm, False)
    else:
        unseen_data_bin = None
    
    if len(seen_data["utter"]) > 0:
        seen_data_bin = binarize_data(seen_data, vocab, params.tgt_dm, False)
    else:
        seen_data_bin = None

    model_path = params.model_path
    assert os.path.isfile(model_path)

    reloaded = torch.load(model_path)
    slu_tagger = reloaded["slu_tagger"]
    slu_tagger.cuda()

    baseline_trainer = BaselineTrainer(params, slu_tagger)

    print("Prepare dataloader ...")
    if unseen_data_bin:
        unseen_dataset = BaselineDataset(unseen_data_bin["utter"], unseen_data_bin["y2"], unseen_data_bin["domains"])

        unseen_dataloader = DataLoader(dataset=unseen_dataset, batch_size=params.batch_size, collate_fn=baseline_collate_fn, shuffle=False)

        _, f1_score, _ = baseline_trainer.evaluate(0, unseen_dataloader, istestset=True)
        print("Evaluate on {} domain unseen slots. Final slot F1 score: {:.4f}.".format(params.tgt_dm, f1_score))

    else:
        print("Number of unseen sample is zero")

    if seen_data_bin:
        seen_dataset = BaselineDataset(seen_data_bin["utter"], seen_data_bin["y2"], seen_data_bin["domains"])
        
        seen_dataloader = DataLoader(dataset=seen_dataset, batch_size=params.batch_size, collate_fn=baseline_collate_fn, shuffle=False)

        _, f1_score, _ = baseline_trainer.evaluate(0, seen_dataloader, istestset=True)
        print("Evaluate on {} domain seen slots. Final slot F1 score: {:.4f}.".format(params.tgt_dm, f1_score))

    else:
        print("Number of seen sample is zero")

if __name__ == "__main__":
    params = get_params()
    if params.model_type == "coach":
        if params.test_mode == "testset":
            test_coach(params)
        elif params.test_mode == "seen_unseen":
            test_coach_on_seen_and_unseen(params)
    else:
        if params.test_mode == "testset":
            test_baseline(params)
        elif params.test_mode == "seen_unseen":
            test_baseline_on_seen_and_unseen(params)
