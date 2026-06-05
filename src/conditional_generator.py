import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class ConditionalMoleculeDataset(Dataset):
    """Dataset that provides (optional protein, optional logP, SMILES).

    Required:
    - "smiles": canonical SMILES string of the ligand.

    Optional (recommended for true conditional generation):
    - protein_column: protein sequence, target name, or target ID
    - logp_column: experimental or predicted logP value (float)

    If protein/logP are not provided, the model effectively trains as an
    unconditional SMILES generator using dummy conditioning.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        smiles_column: str = "smiles",
        protein_column: Optional[str] = None,
        logp_column: Optional[str] = None,
        max_smiles_length: int = 128,
        smiles_tokenizer=None,
    ) -> None:
        if smiles_tokenizer is None:
            raise ValueError("smiles_tokenizer must be provided")

        if smiles_column not in df.columns:
            raise ValueError(f"Missing required SMILES column '{smiles_column}' in DataFrame")

        self.df = df.reset_index(drop=True)
        self.smiles_column = smiles_column

        # Make protein/logP genuinely optional: only use if column exists.
        self.protein_column = protein_column if protein_column in df.columns else None
        self.logp_column = logp_column if logp_column in df.columns else None
        self.max_smiles_length = max_smiles_length
        self.smiles_tokenizer = smiles_tokenizer
        self.unknown_target = "<unk_target>"

        self.target_to_idx: dict[str, int] = {self.unknown_target: 0}
        if self.protein_column is not None:
            target_names = (
                self.df[self.protein_column]
                .fillna(self.unknown_target)
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            for target_name in sorted(target_names):
                if target_name not in self.target_to_idx:
                    self.target_to_idx[target_name] = len(self.target_to_idx)

        # Pre-compute simple statistics for logP normalization if available
        if self.logp_column is not None:
            logp_series = pd.to_numeric(self.df[self.logp_column], errors="coerce").dropna()
            if logp_series.empty:
                self.logp_mean = 0.0
                self.logp_std = 1.0
            else:
                self.logp_mean = float(logp_series.mean())
                std = float(logp_series.std())
                self.logp_std = std if math.isfinite(std) and std != 0.0 else 1.0
        else:
            # Fallback: dummy values so that normalized logP is always 0
            self.logp_mean = 0.0
            self.logp_std = 1.0

    def __len__(self) -> int:
        return len(self.df)

    def normalize_logp(self, value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return (value - self.logp_mean) / self.logp_std

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        smiles = str(row[self.smiles_column])
        # Use a simple placeholder if no protein column is provided
        protein = str(row[self.protein_column]) if self.protein_column is not None else self.unknown_target
        target_id = self.target_to_idx.get(protein, self.target_to_idx[self.unknown_target])
        # Use 0.0 if no logP column is provided (normalizes to 0)
        if self.logp_column is not None:
            raw_logp = pd.to_numeric(pd.Series([row[self.logp_column]]), errors="coerce").iloc[0]
            raw_logp = float(raw_logp) if pd.notna(raw_logp) else 0.0
        else:
            raw_logp = 0.0

        # Tokenize SMILES for decoder
        tok = self.smiles_tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=self.max_smiles_length,
            return_tensors="pt",
        )

        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)
        decoder_input_ids = input_ids[:-1].clone()
        decoder_attention_mask = attention_mask[:-1].clone()
        labels = input_ids[1:].clone()
        labels[attention_mask[1:] == 0] = -100

        item = {
            "input_ids": decoder_input_ids,
            "attention_mask": decoder_attention_mask,
            "labels": labels,
            "protein": protein,
            "target_id": torch.tensor(target_id, dtype=torch.long),
            "logp": torch.tensor(self.normalize_logp(raw_logp), dtype=torch.float32),
        }
        return item


class TargetConditionEncoder(nn.Module):
    """Encodes categorical target IDs into dense representations."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max(vocab_size, 1), embedding_dim)
        self.output_dim = embedding_dim

    def forward(self, target_ids: torch.Tensor) -> torch.Tensor:
        if target_ids.dim() == 0:
            target_ids = target_ids.unsqueeze(0)
        return self.embedding(target_ids.long())


class ProteinConditionEncoder(nn.Module):
    """Encodes a protein sequence into a dense representation.

    For simplicity and robustness, we use a character-level embedding over
    amino-acid residues with a bidirectional GRU. This is sufficient to
    capture secondary-structure-level preferences for ligand design,
    without committing to a specific protein representation library.
    """

    def __init__(
        self,
        vocab: Optional[str] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        if vocab is None:
            # Standard 20 amino acids + common ambiguity / padding tokens
            vocab = "ACDEFGHIKLMNPQRSTVWYXBZJUO-"  # simple, extensible alphabet

        self.vocab = vocab
        self.token_to_idx = {ch: i + 1 for i, ch in enumerate(self.vocab)}
        self.pad_idx = 0

        self.embedding = nn.Embedding(
            num_embeddings=len(self.token_to_idx) + 1,
            embedding_dim=embedding_dim,
            padding_idx=self.pad_idx,
        )

        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.output_dim = 2 * hidden_dim

    def _encode_sequence(self, seq: str) -> torch.Tensor:
        indices = [self.token_to_idx.get(ch, self.pad_idx) for ch in seq]
        if not indices:
            indices = [self.pad_idx]
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        # Convert list of sequences to padded batch of indices
        encoded = [self._encode_sequence(seq) for seq in sequences]
        max_len = max(t.size(0) for t in encoded)
        padded = []
        for t in encoded:
            pad_len = max_len - t.size(0)
            if pad_len > 0:
                t = torch.cat([t, torch.full((pad_len,), self.pad_idx, dtype=torch.long)], dim=0)
            padded.append(t)

        batch = torch.stack(padded, dim=0)  # (batch, seq_len)
        emb = self.embedding(batch)  # (batch, seq_len, emb_dim)
        outputs, _ = self.encoder(emb)  # (batch, seq_len, 2*hidden_dim)
        # Use mean pooling over sequence as global protein descriptor
        protein_repr = outputs.mean(dim=1)
        return protein_repr


class LogPEncoder(nn.Module):
    """Encodes a (normalized) logP scalar into a dense vector."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, logp_values: torch.Tensor) -> torch.Tensor:
        if logp_values.dim() == 1:
            logp_values = logp_values.unsqueeze(-1)
        return self.net(logp_values)


class ConditionalDecoder(nn.Module):
    """A simple Transformer-based decoder conditioned on protein + logP.

    Conditioning is injected as an additive bias into the token embeddings
    (similar to FiLM-style conditioning). This keeps the architecture
    lightweight while still allowing rich control by protein environment
    and physicochemical profile.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Conditioning projection (protein + logP -> additive bias)
        self.condition_proj = nn.Linear(d_model, d_model)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        device = input_ids.device
        batch_size, seq_len = input_ids.size()

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        # Project condition to model dimension and broadcast
        cond_bias = self.condition_proj(condition).unsqueeze(1)  # (batch, 1, d_model)
        hidden_states = token_emb + pos_emb + cond_bias

        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=device)

        # Causal decoder without explicit encoder; condition already injected
        decoded = self.decoder(
            tgt=hidden_states,
            memory=torch.zeros(batch_size, 1, self.d_model, device=device),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None,
        )

        logits = self.lm_head(decoded)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

        return {"loss": loss, "logits": logits}


class ConditionalMoleculeGenerator(nn.Module):
    """High-level conditional model: protein + logP -> SMILES distribution."""

    def __init__(
        self,
        smiles_tokenizer,
        d_model: int = 256,
        condition_mode: str = "target_lookup",
        target_vocab: Optional[dict[str, int]] = None,
        target_embedding_dim: int = 128,
        protein_embedding_dim: int = 128,
        protein_hidden_dim: int = 256,
        logp_dim: int = 128,
    ) -> None:
        super().__init__()

        self.smiles_tokenizer = smiles_tokenizer
        self.condition_mode = condition_mode
        self.target_vocab = target_vocab or {"<unk_target>": 0}
        self.unknown_target = "<unk_target>"

        if self.condition_mode == "target_lookup":
            self.target_encoder = TargetConditionEncoder(
                vocab_size=len(self.target_vocab),
                embedding_dim=target_embedding_dim,
            )
            condition_width = self.target_encoder.output_dim
        else:
            self.protein_encoder = ProteinConditionEncoder(
                embedding_dim=protein_embedding_dim,
                hidden_dim=protein_hidden_dim,
            )
            condition_width = self.protein_encoder.output_dim

        self.logp_encoder = LogPEncoder(output_dim=logp_dim)

        self.condition_fusion = nn.Sequential(
            nn.Linear(condition_width + logp_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.decoder = ConditionalDecoder(
            vocab_size=smiles_tokenizer.vocab_size,
            d_model=d_model,
        )

    def _lookup_target_ids(self, proteins: List[str], device: torch.device) -> torch.Tensor:
        target_ids = [self.target_vocab.get(str(name), self.target_vocab.get(self.unknown_target, 0)) for name in proteins]
        return torch.tensor(target_ids, dtype=torch.long, device=device)

    def encode_condition(
        self,
        proteins: List[str],
        logp_values: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "target_lookup":
            if target_ids is None:
                target_ids = self._lookup_target_ids(proteins, logp_values.device)
            protein_repr = self.target_encoder(target_ids.to(logp_values.device))
        else:
            protein_repr = self.protein_encoder(proteins)
            protein_repr = protein_repr.to(logp_values.device)

        logp_repr = self.logp_encoder(logp_values)
        condition = torch.cat([protein_repr, logp_repr], dim=-1)
        return self.condition_fusion(condition)

    def get_config(self) -> dict[str, Any]:
        return {
            "condition_mode": self.condition_mode,
            "target_vocab": self.target_vocab,
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        proteins: List[str] = batch["protein"]
        target_ids: Optional[torch.Tensor] = batch.get("target_id")
        logp_values: torch.Tensor = batch["logp"]

        condition = self.encode_condition(
            proteins,
            logp_values.to(input_ids.device),
            target_ids=target_ids.to(input_ids.device) if target_ids is not None else None,
        )
        return self.decoder(
            input_ids=input_ids,
            condition=condition,
            attention_mask=attention_mask,
            labels=labels,
        )

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.size(-1):
            return logits
        top_values, _ = torch.topk(logits, k=top_k, dim=-1)
        threshold = top_values[:, -1].unsqueeze(-1)
        return logits.masked_fill(logits < threshold, -1e9)

    @staticmethod
    def _prevent_repeat_runs(logits: torch.Tensor, generated: torch.Tensor, max_repeat_tokens: int) -> torch.Tensor:
        if max_repeat_tokens <= 0 or generated.size(1) < max_repeat_tokens:
            return logits
        recent_tokens = generated[:, -max_repeat_tokens:]
        repeated_rows = (recent_tokens == recent_tokens[:, -1:]).all(dim=1)
        if repeated_rows.any():
            repeated_token_ids = recent_tokens[:, -1]
            logits[repeated_rows, repeated_token_ids[repeated_rows]] = -1e9
        return logits

    @torch.no_grad()
    def generate(
        self,
        proteins: List[str],
        logp_values: List[float],
        max_length: int = 64,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 32,
        min_length: int = 8,
        eos_token_bonus: float = 2.0,
        max_repeat_tokens: int = 3,
    ) -> List[str]:
        """Generate SMILES conditioned on protein sequences and logP values."""

        self.eval()

        device = next(self.parameters()).device
        batch_size = len(proteins)
        logp_tensor = torch.tensor(logp_values, dtype=torch.float32, device=device)
        condition = self.encode_condition(proteins, logp_tensor)

        bos_token_id = self.smiles_tokenizer.bos_token_id
        eos_token_id = self.smiles_tokenizer.eos_token_id
        max_supported_length = int(self.decoder.position_embedding.num_embeddings)
        max_length = min(max_length, max_supported_length)

        input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device,
        )

        finished = [False] * batch_size
        generated = input_ids

        for _ in range(max_length - 1):
            outputs = self.decoder(
                input_ids=generated,
                condition=condition,
            )
            logits = outputs["logits"][:, -1, :] / max(temperature, 1e-5)
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            logits = self._apply_top_k(logits, top_k=top_k)
            logits = self._prevent_repeat_runs(logits, generated=generated, max_repeat_tokens=max_repeat_tokens)

            current_length = generated.size(1)
            if current_length < min_length:
                logits[:, eos_token_id] = -1e9
            else:
                logits[:, eos_token_id] = logits[:, eos_token_id] + eos_token_bonus

            if num_beams == 1:
                probs = torch.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                probs_sum = probs.sum(dim=-1, keepdim=True)
                zero_mass = probs_sum.squeeze(-1) <= 0
                if zero_mass.any():
                    probs[zero_mass] = 1.0 / probs.size(-1)
                    probs_sum = probs.sum(dim=-1, keepdim=True)
                probs = probs / probs_sum
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Simple top-k / beam-like selection (placeholder for real beam search)
                topk = torch.topk(logits, k=num_beams, dim=-1).indices
                next_tokens = topk[:, 0:1]

            generated = torch.cat([generated, next_tokens], dim=-1)

            for i in range(batch_size):
                if not finished[i] and next_tokens[i].item() == eos_token_id:
                    finished[i] = True
            if all(finished):
                break

        smiles_list: List[str] = []
        for seq in generated:
            text = self.smiles_tokenizer.decode(
                seq.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            smiles_list.append(text)

        return smiles_list


@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    num_epochs: int = 10
    max_smiles_length: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ConditionalTrainer:
    """Minimal trainer for the conditional generator.

    This is intentionally lightweight and uses a vanilla PyTorch loop so
    that you can easily customise loss terms (e.g., penalising unrealistic
    logP, adding binding-affinity objectives, etc.).
    """

    def __init__(
        self,
        model: ConditionalMoleculeGenerator,
        dataset: ConditionalMoleculeDataset,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config or TrainingConfig()

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        proteins = [item["protein"] for item in batch]
        target_ids = torch.stack([item["target_id"] for item in batch])
        logp = torch.stack([item["logp"] for item in batch])

        max_active_tokens = int(attention_mask.sum(dim=1).max().item())
        max_active_tokens = max(max_active_tokens, 1)
        input_ids = input_ids[:, :max_active_tokens]
        attention_mask = attention_mask[:, :max_active_tokens]
        labels = labels[:, :max_active_tokens]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "protein": proteins,
            "target_id": target_ids,
            "logp": logp,
        }

    def train(self) -> None:
        device = torch.device(self.config.device)
        self.model.to(device)

        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in self.dataloader:
                batch["input_ids"] = batch["input_ids"].to(device)
                batch["attention_mask"] = batch["attention_mask"].to(device)
                batch["labels"] = batch["labels"].to(device)
                batch["logp"] = batch["logp"].to(device)

                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # Example of how to wire everything together from a DataFrame.
    from transformers import PreTrainedTokenizerFast

    df = pd.read_csv("data/candidates4.csv")  # adjust to your actual file

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer/smiles_tokenizer.json",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    dataset = ConditionalMoleculeDataset(
        df=df,
        smiles_column="smiles",
        protein_column="protein",
        logp_column="logp",
        max_smiles_length=128,
        smiles_tokenizer=tokenizer,
    )

    model = ConditionalMoleculeGenerator(smiles_tokenizer=tokenizer)
    trainer = ConditionalTrainer(model=model, dataset=dataset)
    trainer.train()

    # After training, generate a few candidate molecules for a given target
    example_proteins = [df.iloc[0]["protein"]]
    example_logp = [float(df.iloc[0]["logp"])]
    samples = model.generate(example_proteins, example_logp, max_length=64)
    print("Generated SMILES:", samples)
