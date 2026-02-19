import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from deepspeed.moe.layer import MoE
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

# -----------------------
# Config
# -----------------------
HIDDEN = 256
NUM_HEADS = 4
NUM_LAYERS = 6
VOCAB_SIZE = 50257
SEQ_LEN = 128
NUM_EXPERTS = 4
BATCH_SIZE = 8
LR = 1e-4

# -----------------------
# Transformer Block with MoE FFN
# -----------------------
class TransformerBlock(nn.Module):
    def __init__(self,layer_i, hidden, num_heads, num_experts):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)

        self.ln2 = nn.LayerNorm(hidden)

        # Replace standard FFN with MoE
        self.moe = MoE(
            layer_i,
            hidden_size=hidden,
            expert=nn.Sequential(
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Linear(hidden * 4, hidden),
            ),
            num_experts=num_experts,
            ep_size=1,
            k=1,
            capacity_factor=1.0,
        )

    def forward(self, x):
        # Self Attention
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # MoE Feedforward
        h = self.ln2(x)
        moe_out, _, _ = self.moe(h)
        x = x + moe_out

        return x


# -----------------------
# Minimal GPT with MoE
# -----------------------
class MoEGPT(nn.Module):
    def __init__(self, vocab_size, seq_len, hidden, num_heads, num_layers, num_experts):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)

        self.layers = nn.ModuleList([
            TransformerBlock(i, hidden, num_heads, num_experts)
            for i in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

        self.seq_len = seq_len

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits
    
    def get_communication_matrix(self):
        accumulative_comm_matrix = []
        for layer in self.layers:
            print(f"Layer {layer.moe.deepspeed_moe.layer_i} with communication matrix {layer.moe.deepspeed_moe.comm_matrix_history}")



# -----------------------
# Dataset
# -----------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=SEQ_LEN
    )

dataset = dataset.map(tokenize, remove_columns=["text"])
dataset = dataset.filter(lambda x: len(x["input_ids"]) == SEQ_LEN)
dataset.set_format(type="torch", columns=["input_ids"])

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# -----------------------
# Model + DeepSpeed
# -----------------------
deepspeed.init_distributed()
model = MoEGPT(
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    hidden=HIDDEN,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_experts=NUM_EXPERTS
)

print("[CUSTOM DEBUG] MILESTONE model initialized")

def create_moe_param_groups(model):
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {'params': [p for p in model.parameters()], 'name': 'parameters'}

    return split_params_into_different_moe_groups_for_optimizer(parameters)

param_groups = create_moe_param_groups(model)

optimizer = torch.optim.AdamW(param_groups, lr=LR)

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    model_parameters=param_groups,
    config="escho/ds_minimal_moe_test.json"
)


# -----------------------
# Training Loop
# -----------------------
for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(model.device)

    logits = model(input_ids)
    model.get_communication_matrix()

    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB_SIZE),
        input_ids[:, 1:].reshape(-1)
    )

    model.backward(loss)
    model.step()

    if step % 10 == 0 and step != 0:
        print(f"Step {step}, loss {loss.item():.4f}")
        break


from deepspeed import comm as dist
   
if dist.get_rank() == 0:
    dist.log_summary()

for i, layer in enumerate(model.layers):
    layer.moe.deepspeed_moe.moelens.save(f"moelens_layer{i}.json")