
import numpy as np
import pickle

from src.ner.datareader import datareader
from src.utils import load_embedding

# entity_list = ["LOC", "PER", "ORG", "MISC"]
entity_types = ["location", "person", "organization", "miscellaneous"]  # entity descriptions

example_dict = {
    "location": ["france", "russia"],
    "person": ["quigley", "samokhalova"],
    "organization": ["aberdeen", "nantes"],
    "miscellaneous": ["english", "eu-wide"]
}   # entity examples

def get_oov_words():
    _, _, _, vocab = datareader()
    _ = load_embedding(vocab, 300, "PATH_OF_THE_WIKI_EN_VEC")

def gen_embs_for_vocab():
    _, _, _, vocab = datareader()
    embedding = load_embedding(vocab, 300, "PATH_OF_THE_WIKI_EN_VEC", "../data/ner/emb/oov_embs.txt")

    np.save("../data/ner/emb/ner_embs.npy", embedding)

def gen_embs_for_entity_types(emb_file, emb_dim):
    embedding = np.zeros((len(entity_types), emb_dim))
    print("loading embeddings from %s" % emb_file)
    embedded_words = []
    with open(emb_file, "r") as ef:
        pre_trained = 0
        for i, line in enumerate(ef):
            if i == 0: continue # first line would be "num of words and dimention"
            line = line.strip()
            sp = line.split()
            try:
                assert len(sp) == emb_dim + 1
            except:
                continue
            if sp[0] in entity_types and sp[0] not in embedded_words:
                pre_trained += 1
                embedding[entity_types.index(sp[0])] = [float(x) for x in sp[1:]]
                embedded_words.append(sp[0])
    print("Pre-train: %d / %d (%.2f)" % (pre_trained, len(entity_types), pre_trained / len(entity_types)))
    
    np.save("../data/ner/emb/entity_type_embs.npy", embedding)

def gen_example_embs_for_entity_types(emb_file, emb_dim):
    ner_embs = np.load(emb_file)
    _, _, _, vocab = datareader()
    
    example_embs = np.zeros((len(entity_types), emb_dim, 2))
    for i, entity_type in enumerate(entity_types):
        examples = example_dict[entity_type]
        for j, example in enumerate(examples):
            index = vocab.word2index[example]
            example_embs[i, :, j] = ner_embs[index]
    
    print("saving example embeddings")
    np.save("../data/ner/emb/example_embs.npy", example_embs)

if __name__ == "__main__":
    # get_oov_words()
    # gen_embs_for_vocab()
    # gen_embs_for_entity_types("PATH_OF_THE_WIKI_EN_VEC", 300)
    gen_example_embs_for_entity_types("../data/ner/emb/ner_embs.npy", 300)
    