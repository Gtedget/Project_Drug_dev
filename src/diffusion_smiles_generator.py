from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from conditional_generator import ConditionalMoleculeDataset, LogPEncoder, ProteinConditionEncoder


class DiffusionMoleculeDataset(Dataset):
    def __init__(
        self,
        dataset: ConditionalMoleculeDataset,
    ) -> None:
        self.dataset = dataset
        self.logp_mean = dataset.logp_mean
        self.logp_std = dataset.logp_std

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]


class DiffusionDenoiser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_diffusion_steps: int = 32,
        protein_embedding_dim: int = 64,
        protein_hidden_dim: int = 128,
        logp_dim: int = 64,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.num_diffusion_steps = num_diffusion_steps
        self.protein_encoder = ProteinConditionEncoder(
            embedding_dim=protein_embedding_dim,
            hidden_dim=protein_hidden_dim,
        )
        self.logp_encoder = LogPEncoder(output_dim=logp_dim)
        self.condition_proj = nn.Linear(self.protein_encoder.output_dim + logp_dim, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.timestep_embedding = nn.Embedding(num_diffusion_steps + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode_condition(self, proteins: List[str], logp_values: torch.Tensor) -> torch.Tensor:
        protein_repr = self.protein_encoder(proteins)
        logp_repr = self.logp_encoder(logp_values)
        condition = torch.cat([protein_repr, logp_repr], dim=-1)
        return self.condition_proj(condition)

    def forward(
        self,
        noisy_input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        proteins: List[str],
        logp_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        positions = torch.arange(noisy_input_ids.size(1), device=noisy_input_ids.device).unsqueeze(0)
        hidden_states = self.token_embedding(noisy_input_ids)
        hidden_states = hidden_states + self.position_embedding(positions)
        hidden_states = hidden_states + self.timestep_embedding(timesteps).unsqueeze(1)
        hidden_states = hidden_states + self.encode_condition(proteins, logp_values).unsqueeze(1)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        hidden_states = self.encoder(hidden_states, src_key_padding_mask=key_padding_mask)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


class DiscreteDiffusionSmilesGenerator(nn.Module):
    def __init__(
        self,
        smiles_tokenizer,
        max_length: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_diffusion_steps: int = 32,
        protein_embedding_dim: int = 64,
        protein_hidden_dim: int = 128,
        logp_dim: int = 64,
    ) -> None:
        super().__init__()
        self.smiles_tokenizer = smiles_tokenizer
        self.config = {
            "max_length": max_length,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "num_diffusion_steps": num_diffusion_steps,
            "protein_embedding_dim": protein_embedding_dim,
            "protein_hidden_dim": protein_hidden_dim,
            "logp_dim": logp_dim,
        }
        self.pad_token_id = smiles_tokenizer.pad_token_id
        self.bos_token_id = smiles_tokenizer.bos_token_id
        self.eos_token_id = smiles_tokenizer.eos_token_id
        valid_token_ids = [
            token_id
            for token_id in range(smiles_tokenizer.vocab_size)
            if token_id not in {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        ]
        if not valid_token_ids:
            valid_token_ids = list(range(smiles_tokenizer.vocab_size))
        self.register_buffer("valid_token_ids", torch.tensor(valid_token_ids, dtype=torch.long), persistent=False)
        self.register_buffer(
            "noise_schedule",
            torch.linspace(0.05, 0.6, steps=num_diffusion_steps + 1, dtype=torch.float32),
            persistent=False,
        )
        self.denoiser = DiffusionDenoiser(
            vocab_size=smiles_tokenizer.vocab_size,
            max_length=max_length,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_diffusion_steps=num_diffusion_steps,
            protein_embedding_dim=protein_embedding_dim,
            protein_hidden_dim=protein_hidden_dim,
            logp_dim=logp_dim,
        )

    def sample_random_tokens(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        random_indices = torch.randint(0, self.valid_token_ids.numel(), shape, device=device)
        return self.valid_token_ids.to(device)[random_indices]

    def corrupt_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        device = input_ids.device
        corruption_prob = self.noise_schedule.to(device)[timesteps].unsqueeze(1)
        random_tokens = self.sample_random_tokens(input_ids.shape, device=device)
        corruption_mask = torch.rand(input_ids.shape, device=device) < corruption_prob
        corruption_mask = corruption_mask & attention_mask.bool()
        if self.bos_token_id is not None:
            corruption_mask = corruption_mask & input_ids.ne(self.bos_token_id)
        if self.eos_token_id is not None:
            corruption_mask = corruption_mask & input_ids.ne(self.eos_token_id)
        return torch.where(corruption_mask, random_tokens, input_ids)

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        proteins = batch["protein"]
        logp_values = batch["logp"]
        timesteps = torch.randint(
            1,
            self.config["num_diffusion_steps"] + 1,
            (input_ids.size(0),),
            device=input_ids.device,
        )
        noisy_input_ids = self.corrupt_input_ids(input_ids, attention_mask, timesteps)
        logits = self.denoiser(
            noisy_input_ids=noisy_input_ids,
            timesteps=timesteps,
            proteins=proteins,
            logp_values=logp_values,
            attention_mask=attention_mask,
        )
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            ignore_index=self.pad_token_id,
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        proteins: List[str],
        logp_values: List[float],
        max_length: Optional[int] = None,
        temperature: float = 1.0,
    ) -> List[str]:
        self.eval()
        device = next(self.parameters()).device
        batch_size = len(proteins)
        sequence_length = max_length or self.config["max_length"]
        generated = self.sample_random_tokens(torch.Size([batch_size, sequence_length]), device=device)
        attention_mask = torch.ones_like(generated, device=device)
        logp_tensor = torch.tensor(logp_values, dtype=torch.float32, device=device)

        if self.bos_token_id is not None and sequence_length > 0:
            generated[:, 0] = self.bos_token_id
        if self.eos_token_id is not None and sequence_length > 1:
            generated[:, -1] = self.eos_token_id

        for step in range(self.config["num_diffusion_steps"], 0, -1):
            timesteps = torch.full((batch_size,), step, dtype=torch.long, device=device)
            logits = self.denoiser(
                noisy_input_ids=generated,
                timesteps=timesteps,
                proteins=proteins,
                logp_values=logp_tensor,
                attention_mask=attention_mask,
            )
            probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
            sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(batch_size, sequence_length)
            generated = sampled_tokens
            if self.bos_token_id is not None and sequence_length > 0:
                generated[:, 0] = self.bos_token_id
            if self.eos_token_id is not None and sequence_length > 1:
                generated[:, -1] = self.eos_token_id

        smiles_list: List[str] = []
        for sequence in generated:
            smiles_list.append(
                self.smiles_tokenizer.decode(
                    sequence.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )
        return smiles_list

    def get_config(self) -> Dict[str, Any]:
        return dict(self.config)


@dataclass
class DiffusionTrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    num_epochs: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DiffusionTrainer:
    def __init__(
        self,
        model: DiscreteDiffusionSmilesGenerator,
        dataset: DiffusionMoleculeDataset,
        config: Optional[DiffusionTrainingConfig] = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config or DiffusionTrainingConfig()
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "protein": [item["protein"] for item in batch],
            "logp": torch.stack([item["logp"] for item in batch]),
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
                batch["logp"] = batch["logp"].to(device)

                self.optimizer.zero_grad()
                loss = self.model.compute_loss(batch)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - loss: {avg_loss:.4f}")