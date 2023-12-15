import spacy
import pickle
from tqdm import tqdm
from utils import Data
from collections import namedtuple

if __name__ == "__main__":
    data_processor = Data(root_path="../")
    data_processor.read_in_memory()
    nlp = spacy.load("en_core_web_sm")
    passage_sentence_tokens = {}
    token_tuple = namedtuple(
        "token",
        ["index", "text", "lemma", "pos", "tag", "dep", "parent", "parent_index"],
    )
    for pid, passage in tqdm(data_processor.dataset["collection.sampled"].items()):
        passage_sentence_tokens[pid] = {}
        doc = nlp(passage["passage"])
        for i, sentence in enumerate(doc.sents):
            passage_sentence_tokens[pid][i] = []
            for token in sentence:
                passage_sentence_tokens[pid][i].append(
                    token_tuple(
                        token.i,
                        token.text,
                        token.lemma_,
                        token.pos_,
                        token.tag_,
                        token.dep_,
                        token.head.text,
                        token.head.i,
                    )
                )

    with open("passage_tokens.pkl", "wb") as f:
        pickle.dump(passage_sentence_tokens, f)
