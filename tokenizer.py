import json


# My implementation of Byte Pair Encoding

# https://github.com/openai/gpt-2/blob/master/src/encoder.py

# frequent pairings produce suboptimal behaviour, look at the regex line in the original gpt-2 bpe code
class Tokenizer:

    def __init__(self, vocab_size = 1000):
        self.vocab_size = vocab_size
        self.merges = {}


    def train(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as file:
            dataset = list(file.read().encode("utf-8"))

        tokens = 256
        while (tokens < self.vocab_size):
            counts = {}
            for a, b in zip(dataset, dataset[1:]):
                pair = (a, b)
                counts[pair] = counts.get(pair, 0) + 1

            to_merge = max(counts, key = counts.get)
            self.merges[to_merge] = tokens
            print(f"merged {to_merge} as token number {tokens}")

            dataset = self._merge(dataset, to_merge, tokens)
            tokens += 1

    
    def encode(self, text):
        tokens = list(text.encode("utf-8"))

        while True:
            appearences = {}
            for a, b in zip(tokens, tokens[1:]):
                appearences[(a, b)] = appearences.get((a,b), 0) + 1

            if not appearences:
                break

            pair_to_merge = min(appearences, key=lambda p: self.merges.get(p, float("inf")))
            if pair_to_merge not in self.merges:
                # everything is tokenized to the fullest extent
                break
            idx = self.merges[pair_to_merge]
            tokens = self._merge(tokens, pair_to_merge, idx)

        return tokens
    
    def decode(self, text):
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), i in self.merges.items():
            vocab[i] = vocab[p0] + vocab[p1]
        b = b"".join(vocab[i] for i in text)
        return b.decode('utf-8', errors = "replace")


    def save(self, path):
        to_save = {f"{p0}, {p1}" : i for (p0, p1), i in self.merges.items()}

        with open(path, "w") as file:
            json.dump(to_save, file)
        print(f"{len(self.merges)} merges saved to {path}")

    def load(self, path):
        with open(path, "r") as file:
            raw = json.load(file)

        self.merges = {tuple(p for p in k.split(',')) : v for k, v in raw.items()}

        print(f"{len(self.merges)} merges loaded from {path}")

    
    def _merge(self, ids, pair, new_id):
        i = 0
        new_ids = []
        while i < len(ids):
            if (i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair):
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        
        return new_ids
