import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

SMILES_REGEX = r"\[.*?\]|Br|Cl|Si|Na|Ca|Li|Mg|Al|Sn|Ag|Au|Fe|Zn|[A-Za-z0-9=#\-\+\(\)\\\/ ]"

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Split(
    pattern=SMILES_REGEX,
    behavior="isolated"
)

trainer = trainers.BpeTrainer(
    vocab_size=1000,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
)

tokenizer.train(["data/smiles.txt"], trainer)
tokenizer.save("tokenizer/smiles_tokenizer.json")
print("Tokenizer saved.")
