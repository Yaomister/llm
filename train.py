import os
import torch

from transformer import GPT, Config
from tokenizer import Tokenizer



class Dataloader():
    def __init__(self, batch_size, sequence_length):

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        tokenizer = Tokenizer()
        tokenizer.load("./merges.json")
        
        text = []

        files = [os.path.join("./data", f) for f in os.listdir("./data")]

        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                text.append(f.read())

        tokens = tokenizer.encode("".join(text))
        self.tokens = torch.tensor(tokens)
        self.tokens_count = len(self.tokens)
        self.current_index = 0

    
    def next_batch(self):
        batch_start = self.current_index
        batch_end = batch_start + (self.batch_size * self.sequence_length + 1)
        batch = self.tokens[batch_start : batch_end]
        x = batch[:-1].view(self.batch_size, self.sequence_length)
        y = batch[1:].view(self.batch_size, self.sequence_length)

        self.current_index = batch_end - 1

        if (self.current_index > self.tokens_count):
            self.current_index = 0

        return x, y



if __name__ == "__main__":
    model = GPT(config=Config())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = Dataloader(batch_size=4, sequence_length=32)

    for i in range(100):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        x, y = dataloader.next_batch()
        x = x.to(device) 
        y = y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"epoch {i}, loss: {loss.item()}")

    
