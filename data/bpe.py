import re
import json
from config import Config
from collections import Counter
from argparse import ArgumentParser


class BytePairEncoding:
    def __int__(self):
        pass

    def train(self, text):
        r = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        chunks = r.findall(text)
        chunks = [list(chunk.encode('utf-8')) for chunk in chunks]

        merges = {}

        for new_id in range(256, Config.vocab_size):       
            counts = Counter()      
            for chunk in chunks:
        
                for pair in zip(chunk, chunk[1:]):
                    counts[pair] += 1

            to_merge = max(counts, key=counts.get)

            new_chunks = []
            for chunk in chunks:
                new_chunk = []
                i = 0
                while i < len(chunks):
                    if i + 1 < len(chunks) and chunks[i] == to_merge[0] and chunk[i + 1] == to_merge[1]:
                        new_chunk.append(new_id)
                        i += 2
                    else:
                        new_chunk.append(chunk[i])
                        i += 1
                new_chunks.append(new_chunk)

            chunks = new_chunks
            merges[to_merge] = new_id

            self._save(merges)

    def _save(merges):
        with open("/data/merges.json", "w") as f:
            f.write(json.dumps(merges))

            
if __name__ == "__main__":

    bpe = BytePairEncoding()

    with open("data/datasets/text.txt", "r") as f:
        text = f.read()

    bpe.train(text)
     
