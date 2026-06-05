from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class ConditionalGraphMoleculeDataset(Dataset):
    """Dataset providing (optional protein, optional logP, SMILES) for GNN model.

    Required:
    - "smiles": canonical ligand SMILES.

    Optional (recommended for conditioning):
    - protein_column: protein sequence / identifier / target name
    - logp_column: experimental or predicted logP

    If protein/logP are absent, the model behaves as an unconditional
    SMILES generator with dummy conditioning.
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

        self.protein_column = protein_column if protein_column in df.columns else None
        self.logp_column = logp_column if logp_column in df.columns else None
        self.max_smiles_length = max_smiles_length
        self.smiles_tokenizer = smiles_tokenizer

        if self.logp_column is not None:
            logp_series = self.df[self.logp_column].astype(float)
            self.logp_mean = float(logp_series.mean())
            self.logp_std = float(logp_series.std() or 1.0)
        else:
            self.logp_mean = 0.0
            self.logp_std = 1.0

    def __len__(self) -> int:
        return len(self.df)

    def normalize_logp(self, value: float) -> float:
        return (value - self.logp_mean) / self.logp_std

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        smiles = str(row[self.smiles_column])
        protein = str(row[self.protein_column]) if self.protein_column is not None else "UNK_TARGET"
        raw_logp = float(row[self.logp_column]) if self.logp_column is not None else 0.0

        tok = self.smiles_tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=self.max_smiles_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "labels": tok["input_ids"].squeeze(0),
            "protein": protein,
            "logp": torch.tensor(self.normalize_logp(raw_logp), dtype=torch.float32),
        }
        return item


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention over a dense adjacency matrix."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.attn = nn.Linear(2 * output_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        num_nodes = h.size(0)

        h_i = h.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        h_j = h.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        attn_inputs = torch.cat([h_i, h_j], dim=-1)
        attn_logits = self.leaky_relu(self.attn(attn_inputs).squeeze(-1))

        masked_logits = attn_logits.masked_fill(adj <= 0, float("-inf"))
        attn_weights = torch.softmax(masked_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.relu(attn_weights @ h)


class GraphProteinEncoder(nn.Module):
    """Graph encoder over a residue graph.

    - Nodes: amino-acid residues (from the sequence)
    - Edges: simple chain connectivity i <-> i+1 (can be extended to contact maps)

    Supports either a lightweight GCN-style message passing stack or a
    graph attention network (GAT) implemented in pure PyTorch.
    """

    def __init__(
        self,
        vocab: Optional[str] = None,
        node_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        encoder_type: str = "gat",
        gat_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if vocab is None:
            vocab = "ACDEFGHIKLMNPQRSTVWYXBZJUO-"  # standard AA alphabet + extras

        self.vocab = vocab
        self.token_to_idx = {ch: i + 1 for i, ch in enumerate(self.vocab)}
        self.pad_idx = 0

        self.node_embedding = nn.Embedding(
            num_embeddings=len(self.token_to_idx) + 1,
            embedding_dim=node_dim,
            padding_idx=self.pad_idx,
        )

        encoder_type = encoder_type.lower()
        if encoder_type not in {"gcn", "gat"}:
            raise ValueError("encoder_type must be either 'gcn' or 'gat'")
        self.encoder_type = encoder_type

        self.layers = nn.ModuleList()
        input_dim = node_dim
        for _ in range(num_layers):
            if self.encoder_type == "gat":
                self.layers.append(GraphAttentionLayer(input_dim, hidden_dim, dropout=gat_dropout))
            else:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.output_dim = hidden_dim

    def _sequence_to_graph(self, seq: str) -> torch.Tensor:
        indices = [self.token_to_idx.get(ch, self.pad_idx) for ch in seq]
        if not indices:
            indices = [self.pad_idx]
        return torch.tensor(indices, dtype=torch.long)

    @staticmethod
    def _build_chain_adjacency(num_nodes: int, device: torch.device) -> torch.Tensor:
        # Simple chain graph: i <-> i+1, with self loops
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
        for i in range(num_nodes):
            adj[i, i] = 1.0
            if i + 1 < num_nodes:
                adj[i, i + 1] = 1.0
                adj[i + 1, i] = 1.0
        return adj

    @staticmethod
    def _graph_conv(x: torch.Tensor, adj: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        # Degree-normalised adjacency: D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=-1)  # (N,)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        x = adj_norm @ x
        x = layer(x)
        x = torch.relu(x)
        return x

    def encode_single(self, seq: str, device: torch.device) -> torch.Tensor:
        node_indices = self._sequence_to_graph(seq).to(device)
        x = self.node_embedding(node_indices)  # (N, node_dim)
        adj = self._build_chain_adjacency(num_nodes=x.size(0), device=device)

        for layer in self.layers:
            if self.encoder_type == "gat":
                x = layer(x, adj)
            else:
                x = self._graph_conv(x, adj, layer)

        # Graph-level representation via mean pooling over nodes
        return x.mean(dim=0)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        reps = [self.encode_single(seq, device=device) for seq in sequences]
        return torch.stack(reps, dim=0)


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
    """Transformer decoder conditioned on GNN-based protein + logP embedding."""

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
        self.condition_proj = nn.Linear(d_model, d_model)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
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

        cond_bias = self.condition_proj(condition).unsqueeze(1)
        hidden_states = token_emb + pos_emb + cond_bias

        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=device)

        decoded = self.decoder(
            tgt=hidden_states,
            memory=torch.zeros(batch_size, 1, self.d_model, device=device),
            tgt_mask=tgt_mask,
        )

        logits = self.lm_head(decoded)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return {"loss": loss, "logits": logits}


class ConditionalGNNSmilesGenerator(nn.Module):
    """Conditional SMILES generator using a graph neural network for proteins.

    Conditions:
    - Protein (sequence -> residue graph -> GNN encoder)
    - logP (scalar -> MLP encoder)

    Decoding is done at the SMILES token level, ensuring compatibility with
    your existing tokenizer and post-processing pipeline.
    """

    def __init__(
        self,
        smiles_tokenizer,
        d_model: int = 256,
        protein_node_dim: int = 128,
        protein_hidden_dim: int = 256,
        logp_dim: int = 128,
        protein_encoder_type: str = "gat",
        gat_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.smiles_tokenizer = smiles_tokenizer
        self.config = {
            "d_model": d_model,
            "protein_node_dim": protein_node_dim,
            "protein_hidden_dim": protein_hidden_dim,
            "logp_dim": logp_dim,
            "protein_encoder_type": protein_encoder_type,
            "gat_dropout": gat_dropout,
        }

        self.protein_encoder = GraphProteinEncoder(
            node_dim=protein_node_dim,
            hidden_dim=protein_hidden_dim,
            encoder_type=protein_encoder_type,
            gat_dropout=gat_dropout,
        )

        self.logp_encoder = LogPEncoder(output_dim=logp_dim)

        self.condition_fusion = nn.Sequential(
            nn.Linear(self.protein_encoder.output_dim + logp_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.decoder = ConditionalDecoder(
            vocab_size=smiles_tokenizer.vocab_size,
            d_model=d_model,
        )

    def encode_condition(self, proteins: List[str], logp_values: torch.Tensor) -> torch.Tensor:
        protein_repr = self.protein_encoder(proteins)
        logp_repr = self.logp_encoder(logp_values)
        condition = torch.cat([protein_repr, logp_repr], dim=-1)
        return self.condition_fusion(condition)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        proteins: List[str] = batch["protein"]
        logp_values: torch.Tensor = batch["logp"]

        condition = self.encode_condition(proteins, logp_values.to(input_ids.device))
        return self.decoder(
            input_ids=input_ids,
            condition=condition,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        proteins: List[str],
        logp_values: List[float],
        max_length: int = 128,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> List[List[str]]:
        """Generate SMILES for each (protein, logP) pair.

        Returns a list of lists, one inner list of SMILES per condition.
        """

        self.eval()
        device = next(self.parameters()).device

        all_results: List[List[str]] = []

        for protein, logp in zip(proteins, logp_values):
            cond_logp = torch.tensor([logp], dtype=torch.float32, device=device)
            condition = self.encode_condition([protein], cond_logp)

            bos_id = self.smiles_tokenizer.bos_token_id
            eos_id = self.smiles_tokenizer.eos_token_id

            samples: List[str] = []
            for _ in range(num_samples):
                input_ids = torch.full(
                    (1, 1),
                    bos_id,
                    dtype=torch.long,
                    device=device,
                )

                finished = False
                generated = input_ids

                for _ in range(max_length - 1):
                    outputs = self.decoder(
                        input_ids=generated,
                        condition=condition,
                    )
                    logits = outputs["logits"][:, -1, :] / max(temperature, 1e-5)
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=-1)
                    if next_token.item() == eos_id:
                        finished = True
                        break

                text = self.smiles_tokenizer.decode(
                    generated[0].tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                samples.append(text)

            all_results.append(samples)

        return all_results

    def get_config(self) -> Dict[str, Any]:
        return dict(self.config)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    num_epochs: int = 10
    max_smiles_length: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GNNConditionalTrainer:
    """Trainer for the GNN-conditioned SMILES generator."""

    def __init__(
        self,
        model: ConditionalGNNSmilesGenerator,
        dataset: ConditionalGraphMoleculeDataset,
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
        logp = torch.stack([item["logp"] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "protein": proteins,
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
    # Example wiring: train and sample from the GNN-conditioned generator.
    from transformers import PreTrainedTokenizerFast

    df = pd.read_csv("data/candidates4.csv")  # adjust to your actual file

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizer/smiles_tokenizer.json",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    dataset = ConditionalGraphMoleculeDataset(
        df=df,
        smiles_column="smiles",
        protein_column="protein",
        logp_column="logp",
        max_smiles_length=128,
        smiles_tokenizer=tokenizer,
    )

    model = ConditionalGNNSmilesGenerator(smiles_tokenizer=tokenizer)
    trainer = GNNConditionalTrainer(model=model, dataset=dataset)
    trainer.train()

    # Generate a few samples for a given target protein and logP
    example_proteins = [df.iloc[0]["protein"]]
    example_logp = [float(df.iloc[0]["logp"])]
    samples = model.generate(example_proteins, example_logp, max_length=64, num_samples=5)
    print("Generated SMILES for first protein target:")
    for s in samples[0]:
        print(" ", s)
