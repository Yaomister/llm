import os
import math
import time
import torch
from tokenizer import Tokenizer
from transformer import GPT, Config
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group




ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    # the use of DDP needs CUDA
    assert torch.cuda.is_available(), "DDP requires CUDA for now"
    # initialize distribution backend
    init_process_group(backend= "nccl")
    # the rank is which GPU overall
    ddp_rank = int(os.environ["RANK"])
    # the local_rank is which GPU on this machine
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else: # this is for a non ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"using device: {device}")


class Dataloader():
    def __init__(self, batch_size, sequence_length, process_rank, num_processes):

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes

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
        self.current_position = self.batch_size * self.sequence_length * self.process_rank

    def next_batch(self):
        batch_size, sequence_length = self.batch_size, self.sequence_length
        # grab a chunk of size B * T + 1
        buf = self.tokens[self.current_position:self.current_position + batch_size * sequence_length + 1]
        x = buf[:-1].view(batch_size, sequence_length)
        y = buf[1:].view(batch_size, sequence_length)
        self.current_position += batch_size * sequence_length + 1
        if (self.current_position + (batch_size * sequence_length + 1) >=  len(self.tokens)):
            # reset position to beginning of data, w.r.t. process rank
            self.current_position = batch_size * sequence_length * self.process_rank
        

        return x, y


"""
The original GPT-3 paper trains on batches of 500k tokens, which would cause my laptop to expload if I tried to run it, so we're
gonna do batches of gradient accumulation.
"""

total_batch_size = 524288
batch_size = 4
sequence_length = 1024
assert total_batch_size % (batch_size  * sequence_length * ddp_world_size) == 0, "dimensions must match"
# we need to multiply by ddp_world_size because we are running processes on multiple GPU and all of those gradients should be accounted for for a batch
gradient_accumulation_steps = total_batch_size // (batch_size * sequence_length * ddp_world_size)
 

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
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = Dataloader(batch_size=batch_size, sequence_length=sequence_length)

    optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=max_learning_rate)

    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        total_loss = 0.0
        # gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            x, y = dataloader.next_batch()
            x = x.to(device) 
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(x, y)
            
            # we need to normalize the microbatch before summing because 7:24 PMClaude responded: PyTorch takes the mean of the loss within each micro-batch, so accumulating 32 of those gives you a sum of means rather than the mean of the whole batch.PyTorch takes the mean of the loss within each micro-batch
            loss = loss / gradient_accumulation_steps 
            total_loss += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
             # This accumulates because we dont clear the gradients in the inner loop
            loss.backward()
        if ddp:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.AVG)
            

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
        tokens_per_sec = (batch_size * sequence_length) / (t1 - t0)
        print(f"step {step:4d} | loss: {loss.item():.6f} | lr: {learning_rate:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

    if ddp:
        destroy_process_group() # DDP cleanup
    
    torch.save(model.state_dict(), "model.pt")
    

