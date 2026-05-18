import os
import json
import math
import time
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

"""
The original GPT-3 paper trains on batches of 500k tokens, which would cause my laptop to expload if I tried to run it, so we're
gonna do batches of gradient accumulation.
"""

total_batch_size = 524288
batch_size = 16
sequence_length = 1024
assert total_batch_size % (batch_size  * sequence_length) == 0, "dimensions must match"
gradient_accumulation_steps = total_batch_size // (batch_size * sequence_length)
 

max_learning_rate = 6e-4 # according to GPT-3 paper
min_learning_rate = max_learning_rate * 0.1
warm_up_steps = 10
max_steps = 50
def get_learning_rate(it):
    if it < warm_up_steps:
        return max_learning_rate * (it  + 1)/ warm_up_steps
    elif it > max_steps:
        return min_learning_rate
    decay_ratio = (it - warm_up_steps) / (max_steps - warm_up_steps)
    assert (0 <= decay_ratio <= 1)
    factor = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate * factor * (max_learning_rate - min_learning_rate)



if __name__ == "__main__":
    torch.set_float32_matmul_precision(precision="high")
    model = GPT(config=Config())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = Dataloader(batch_size=batch_size, sequence_length=sequence_length)

    for step in range(max_steps):
        t0 = time.time()
        optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=max_learning_rate, device=device)
        optimizer.zero_grad()

        # gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            x, y = dataloader.next_batch()
            x = x.to(device) 
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(x, y)
            loss.backward()

        # clipping the global norm at 1
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        learning_rate = get_learning_rate(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        optimizer.step()
        # wait for GPU to finish the scheduled work above
        torch.cuda.synchronize()

        t1 = time.time()
        dt = (t0 - t1)
        tokens_per_sec = (dataloader.B * dataloader.T) / (t1 - t0)
        print(f"step {step:4d} | loss: {loss.item():.6f} | lr: {learning_rate:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

    
    torch.save(model.state_dict(), "model.pt")
    
