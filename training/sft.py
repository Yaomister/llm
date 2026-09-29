from utils.checkpoint import load_checkpoint





def sft():
    model_data, optimizer_data, meta_data = load_checkpoint("")

if __name__ == "__main__":
    sft()