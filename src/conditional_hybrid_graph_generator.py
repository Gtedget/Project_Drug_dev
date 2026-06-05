import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_DESCRIPTOR_NAMES: Sequence[str] = (
    "hba",
    "hbd",
    "psa",
    "rtb",
    "mol_wt",
    "aromatic_rings",
    "heavy_atoms",
    "qed",
    "fraction_csp3",
    "ring_count",
)

DATAFRAME_DESCRIPTOR_COLUMNS = {
    "hba": "hba",
    "hbd": "hbd",
    "psa": "psa",
    "rtb": "rtb",
    "mol_wt": "full_mwt",
    "aromatic_rings": "aromatic_rings",
    "heavy_atoms": "heavy_atoms",
    "qed": "qed_weighted",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _tokenize_text(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return tokens or ["unk"]


def _get_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def compute_descriptor_value(name: str, smiles: str, row: Optional[pd.Series] = None) -> float:
    if row is not None and name in DATAFRAME_DESCRIPTOR_COLUMNS:
        column = DATAFRAME_DESCRIPTOR_COLUMNS[name]
        if column in row.index and not pd.isna(row[column]):
            return _safe_float(row[column])

    mol = _get_mol(smiles)
    if mol is None:
        return 0.0

    if name == "hba":
        return float(Lipinski.NumHAcceptors(mol))
    if name == "hbd":
        return float(Lipinski.NumHDonors(mol))
    if name == "psa":
        return float(rdMolDescriptors.CalcTPSA(mol))
    if name == "rtb":
        return float(Lipinski.NumRotatableBonds(mol))
    if name == "mol_wt":
        return float(Descriptors.MolWt(mol))
    if name == "aromatic_rings":
        return float(rdMolDescriptors.CalcNumAromaticRings(mol))
    if name == "heavy_atoms":
        return float(mol.GetNumHeavyAtoms())
    if name == "qed":
        return float(QED.qed(mol))
    if name == "fraction_csp3":
        return float(rdMolDescriptors.CalcFractionCSP3(mol))
    if name == "ring_count":
        return float(rdMolDescriptors.CalcNumRings(mol))

    return 0.0


def build_descriptor_vector(
    descriptor_names: Sequence[str],
    smiles: str,
    row: Optional[pd.Series] = None,
) -> List[float]:
    return [compute_descriptor_value(name, smiles=smiles, row=row) for name in descriptor_names]


class ConditionalHybridMoleculeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        smiles_column: str = "smiles",
        structure_smiles_column: Optional[str] = None,
        protein_column: Optional[str] = None,
        binding_site_column: Optional[str] = None,
        logp_column: Optional[str] = None,
        descriptor_names: Optional[Sequence[str]] = None,
        max_smiles_length: int = 128,
        smiles_tokenizer=None,
    ) -> None:
        if smiles_tokenizer is None:
            raise ValueError("smiles_tokenizer must be provided")
        if smiles_column not in df.columns:
            raise ValueError(f"Missing required SMILES column '{smiles_column}' in DataFrame")

        self.df = df.reset_index(drop=True)
        self.smiles_column = smiles_column
        self.structure_smiles_column = structure_smiles_column if structure_smiles_column in df.columns else smiles_column
        self.protein_column = protein_column if protein_column in df.columns else None
        self.binding_site_column = binding_site_column if binding_site_column in df.columns else None
        self.logp_column = logp_column if logp_column in df.columns else None
        self.max_smiles_length = max_smiles_length
        self.smiles_tokenizer = smiles_tokenizer
        self.descriptor_names = list(descriptor_names or DEFAULT_DESCRIPTOR_NAMES)
        self.target_graphs: list[dict[str, torch.Tensor]] = []
        self.ligand_graphs: list[dict[str, torch.Tensor]] = []

        if self.logp_column is not None:
            logp_series = self.df[self.logp_column].map(_safe_float)
            self.logp_mean = float(logp_series.mean())
            self.logp_std = float(logp_series.std() or 1.0)
        else:
            self.logp_mean = 0.0
            self.logp_std = 1.0

        descriptor_rows = []
        for _, row in self.df.iterrows():
            structure_smiles = str(row[self.structure_smiles_column])
            protein = str(row[self.protein_column]) if self.protein_column is not None else "UNK_TARGET"
            binding_site = str(row[self.binding_site_column]) if self.binding_site_column is not None and not pd.isna(row[self.binding_site_column]) else ""
            descriptor_rows.append(build_descriptor_vector(self.descriptor_names, smiles=structure_smiles, row=row))
            self.target_graphs.append(TargetPocketGraphEncoder.build_graph_data(protein=protein, binding_site=binding_site))
            self.ligand_graphs.append(LigandGraphEncoder.mol_to_graph_data(smiles=structure_smiles))

        descriptor_tensor = torch.tensor(descriptor_rows, dtype=torch.float32)
        if descriptor_tensor.numel() == 0:
            descriptor_tensor = torch.zeros((len(self.df), len(self.descriptor_names)), dtype=torch.float32)
        self.descriptor_mean = descriptor_tensor.mean(dim=0)
        self.descriptor_std = descriptor_tensor.std(dim=0)
        self.descriptor_std[self.descriptor_std == 0] = 1.0
        self.normalized_descriptors = (descriptor_tensor - self.descriptor_mean) / self.descriptor_std
        self.descriptor_dim = len(self.descriptor_names)

    def __len__(self) -> int:
        return len(self.df)

    def normalize_logp(self, value: float) -> float:
        return (value - self.logp_mean) / self.logp_std

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        smiles = str(row[self.smiles_column])
        structure_smiles = str(row[self.structure_smiles_column])
        protein = str(row[self.protein_column]) if self.protein_column is not None else "UNK_TARGET"
        binding_site = str(row[self.binding_site_column]) if self.binding_site_column is not None and not pd.isna(row[self.binding_site_column]) else ""
        raw_logp = _safe_float(row[self.logp_column]) if self.logp_column is not None else 0.0

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

        return {
            "input_ids": decoder_input_ids,
            "attention_mask": decoder_attention_mask,
            "labels": labels,
            "protein": protein,
            "binding_site": binding_site,
            "seed_smiles": structure_smiles,
            "logp": torch.tensor(self.normalize_logp(raw_logp), dtype=torch.float32),
            "descriptor_vector": self.normalized_descriptors[idx],
            "target_graph": self.target_graphs[idx],
            "ligand_graph": self.ligand_graphs[idx],
        }


class GraphAttentionLayer(nn.Module):
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


class TargetPocketGraphEncoder(nn.Module):
    def __init__(
        self,
        token_vocab_size: int = 8192,
        node_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        encoder_type: str = "gat",
        gat_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_vocab_size = token_vocab_size
        self.token_embedding = nn.Embedding(token_vocab_size, node_dim)
        self.segment_embedding = nn.Embedding(2, node_dim)

        self.encoder_type = encoder_type
        self.layers = nn.ModuleList()
        input_dim = node_dim
        for _ in range(num_layers):
            if encoder_type == "gat":
                self.layers.append(GraphAttentionLayer(input_dim, hidden_dim, dropout=gat_dropout))
            else:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.output_dim = hidden_dim

    def _hash_token(self, token: str) -> int:
        return abs(hash(token)) % self.token_vocab_size

    @staticmethod
    def _hash_token_static(token: str, token_vocab_size: int) -> int:
        return abs(hash(token)) % token_vocab_size

    @staticmethod
    def _tokenize_graph_inputs(protein: str, binding_site: str) -> tuple[List[str], List[str]]:
        protein_tokens = _tokenize_text(protein)
        pocket_tokens = _tokenize_text(binding_site) if binding_site.strip() else []

        return protein_tokens, pocket_tokens

    def _build_graph(self, protein: str, binding_site: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_data = self.build_graph_data(protein=protein, binding_site=binding_site)
        return (
            graph_data["token_ids"].to(device),
            graph_data["segment_ids"].to(device),
            graph_data["adj"].to(device),
        )

    @classmethod
    def build_graph_data(
        cls,
        protein: str,
        binding_site: str,
        token_vocab_size: int = 8192,
    ) -> dict[str, torch.Tensor]:
        protein_tokens, pocket_tokens = cls._tokenize_graph_inputs(protein, binding_site)

        tokens = protein_tokens + pocket_tokens
        if not tokens:
            tokens = ["unk"]
        token_ids = torch.tensor([cls._hash_token_static(token, token_vocab_size) for token in tokens], dtype=torch.long)
        segment_ids = torch.tensor([0] * len(protein_tokens) + [1] * len(pocket_tokens), dtype=torch.long)
        if segment_ids.numel() == 0:
            segment_ids = torch.zeros((1,), dtype=torch.long)

        num_nodes = len(tokens)
        adj = torch.eye(num_nodes, dtype=torch.float32)

        for i in range(max(len(protein_tokens) - 1, 0)):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
            if i + 2 < len(protein_tokens):
                adj[i, i + 2] = 1.0
                adj[i + 2, i] = 1.0

        pocket_offset = len(protein_tokens)
        for i in range(max(len(pocket_tokens) - 1, 0)):
            left = pocket_offset + i
            right = pocket_offset + i + 1
            adj[left, right] = 1.0
            adj[right, left] = 1.0

        protein_index = {}
        for idx, token in enumerate(protein_tokens):
            protein_index.setdefault(token, []).append(idx)

        shared_edges = 0
        for pocket_idx, token in enumerate(pocket_tokens):
            for protein_idx in protein_index.get(token, []):
                left = protein_idx
                right = pocket_offset + pocket_idx
                adj[left, right] = 1.0
                adj[right, left] = 1.0
                shared_edges += 1

        if pocket_tokens and shared_edges == 0:
            adj[0, pocket_offset] = 1.0
            adj[pocket_offset, 0] = 1.0

        return {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "adj": adj,
        }

    @staticmethod
    def _graph_conv(x: torch.Tensor, adj: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        deg = adj.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        d_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
        return torch.relu(layer(adj_norm @ x))

    def encode_single(self, protein: str, binding_site: str, device: torch.device) -> torch.Tensor:
        token_ids, segment_ids, adj = self._build_graph(protein, binding_site, device=device)
        return self.encode_graph(token_ids=token_ids, segment_ids=segment_ids, adj=adj)

    def encode_graph(self, token_ids: torch.Tensor, segment_ids: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids) + self.segment_embedding(segment_ids)
        for layer in self.layers:
            if self.encoder_type == "gat":
                x = layer(x, adj)
            else:
                x = self._graph_conv(x, adj, layer)
        return x.mean(dim=0)

    def forward(
        self,
        proteins: List[str],
        binding_sites: List[str],
        graph_batch: Optional[List[dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if graph_batch is not None:
            reps = [
                self.encode_graph(
                    token_ids=graph_data["token_ids"].to(device),
                    segment_ids=graph_data["segment_ids"].to(device),
                    adj=graph_data["adj"].to(device),
                )
                for graph_data in graph_batch
            ]
        else:
            reps = [self.encode_single(protein, binding_site, device) for protein, binding_site in zip(proteins, binding_sites)]
        return torch.stack(reps, dim=0)


class DescriptorEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, descriptor_values: torch.Tensor) -> torch.Tensor:
        return self.net(descriptor_values)


class LogPEncoder(nn.Module):
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


class LigandGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        encoder_type: str = "gat",
        gat_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if encoder_type == "gat":
                self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, dropout=gat_dropout))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_dim = hidden_dim

    @staticmethod
    def _atom_features(atom: Chem.Atom) -> List[float]:
        hybridization = float(int(atom.GetHybridization()))
        chiral_tag = float(int(atom.GetChiralTag()))
        return [
            atom.GetAtomicNum() / 100.0,
            atom.GetTotalDegree() / 6.0,
            atom.GetFormalCharge() / 4.0,
            atom.GetTotalNumHs(includeNeighbors=True) / 4.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            hybridization / 6.0,
            chiral_tag / 4.0,
        ]

    @staticmethod
    def mol_to_graph_data(smiles: str) -> dict[str, torch.Tensor]:
        mol = _get_mol(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            x = torch.zeros((1, 8), dtype=torch.float32)
            adj = torch.eye(1, dtype=torch.float32)
            return {"features": x, "adj": adj}

        features = torch.tensor([LigandGraphEncoder._atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
        adj = torch.eye(mol.GetNumAtoms(), dtype=torch.float32)
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            adj[begin, end] = 1.0
            adj[end, begin] = 1.0
        return {"features": features, "adj": adj}

    @staticmethod
    def _mol_to_graph(smiles: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        graph_data = LigandGraphEncoder.mol_to_graph_data(smiles)
        return graph_data["features"].to(device), graph_data["adj"].to(device)

    @staticmethod
    def _graph_conv(x: torch.Tensor, adj: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        deg = adj.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        d_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
        return torch.relu(layer(adj_norm @ x))

    def encode_single(self, smiles: str, device: torch.device) -> torch.Tensor:
        x, adj = self._mol_to_graph(smiles, device=device)
        return self.encode_graph(features=x, adj=adj)

    def encode_graph(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = features
        x = torch.relu(self.input_proj(x))
        for layer in self.layers:
            if self.encoder_type == "gat":
                x = layer(x, adj)
            else:
                x = self._graph_conv(x, adj, layer)
        return x.mean(dim=0)

    def forward(self, smiles_list: List[str], graph_batch: Optional[List[dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        if graph_batch is not None:
            reps = [
                self.encode_graph(
                    features=graph_data["features"].to(device),
                    adj=graph_data["adj"].to(device),
                )
                for graph_data in graph_batch
            ]
        else:
            reps = [self.encode_single(smiles, device) for smiles in smiles_list]
        return torch.stack(reps, dim=0)


class ConditionalDecoder(nn.Module):
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
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

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
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions) + self.condition_proj(condition).unsqueeze(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=device)
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
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return {"loss": loss, "logits": logits}


class ConditionalGNNSmilesGenerator(nn.Module):
    def __init__(
        self,
        smiles_tokenizer,
        d_model: int = 256,
        text_node_dim: int = 128,
        text_hidden_dim: int = 256,
        ligand_hidden_dim: int = 256,
        descriptor_input_dim: int = len(DEFAULT_DESCRIPTOR_NAMES),
        descriptor_hidden_dim: int = 128,
        logp_dim: int = 64,
        protein_encoder_type: str = "gat",
        gat_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.smiles_tokenizer = smiles_tokenizer
        self.config = {
            "d_model": d_model,
            "text_node_dim": text_node_dim,
            "text_hidden_dim": text_hidden_dim,
            "ligand_hidden_dim": ligand_hidden_dim,
            "descriptor_input_dim": descriptor_input_dim,
            "descriptor_hidden_dim": descriptor_hidden_dim,
            "logp_dim": logp_dim,
            "protein_encoder_type": protein_encoder_type,
            "gat_dropout": gat_dropout,
        }

        self.target_pocket_encoder = TargetPocketGraphEncoder(
            node_dim=text_node_dim,
            hidden_dim=text_hidden_dim,
            encoder_type=protein_encoder_type,
            gat_dropout=gat_dropout,
        )
        self.ligand_encoder = LigandGraphEncoder(
            hidden_dim=ligand_hidden_dim,
            encoder_type=protein_encoder_type,
            gat_dropout=gat_dropout,
        )
        self.descriptor_encoder = DescriptorEncoder(descriptor_input_dim, descriptor_hidden_dim)
        self.logp_encoder = LogPEncoder(output_dim=logp_dim)

        fused_dim = (
            self.target_pocket_encoder.output_dim
            + self.ligand_encoder.output_dim
            + descriptor_hidden_dim
            + logp_dim
        )
        self.condition_fusion = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.decoder = ConditionalDecoder(vocab_size=smiles_tokenizer.vocab_size, d_model=d_model)

    def encode_condition(
        self,
        proteins: List[str],
        binding_sites: List[str],
        logp_values: torch.Tensor,
        descriptor_vectors: torch.Tensor,
        seed_smiles: List[str],
        target_graphs: Optional[List[dict[str, torch.Tensor]]] = None,
        ligand_graphs: Optional[List[dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        target_repr = self.target_pocket_encoder(proteins, binding_sites, graph_batch=target_graphs)
        ligand_repr = self.ligand_encoder(seed_smiles, graph_batch=ligand_graphs)
        descriptor_repr = self.descriptor_encoder(descriptor_vectors)
        logp_repr = self.logp_encoder(logp_values)
        condition = torch.cat([target_repr, ligand_repr, descriptor_repr, logp_repr], dim=-1)
        return self.condition_fusion(condition)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = batch["input_ids"]
        condition = self.encode_condition(
            proteins=batch["protein"],
            binding_sites=batch.get("binding_site", [""] * input_ids.size(0)),
            logp_values=batch["logp"].to(input_ids.device),
            descriptor_vectors=batch["descriptor_vector"].to(input_ids.device),
            seed_smiles=batch.get("seed_smiles", [""] * input_ids.size(0)),
            target_graphs=batch.get("target_graph"),
            ligand_graphs=batch.get("ligand_graph"),
        )
        return self.decoder(
            input_ids=input_ids,
            condition=condition,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

    @torch.no_grad()
    def generate(
        self,
        proteins: List[str],
        logp_values: List[float],
        binding_sites: Optional[List[str]] = None,
        descriptor_vectors: Optional[List[List[float]]] = None,
        seed_smiles: Optional[List[str]] = None,
        max_length: int = 128,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> List[List[str]]:
        self.eval()
        device = next(self.parameters()).device
        binding_sites = binding_sites or [""] * len(proteins)
        seed_smiles = seed_smiles or [""] * len(proteins)

        if descriptor_vectors is None:
            descriptor_vectors = [[0.0] * self.config["descriptor_input_dim"] for _ in proteins]

        all_results: List[List[str]] = []
        for protein, binding_site, logp, descriptor_vector, seed_smiles_value in zip(
            proteins,
            binding_sites,
            logp_values,
            descriptor_vectors,
            seed_smiles,
        ):
            cond_logp = torch.tensor([logp], dtype=torch.float32, device=device)
            cond_descriptor = torch.tensor([descriptor_vector], dtype=torch.float32, device=device)
            condition = self.encode_condition(
                proteins=[protein],
                binding_sites=[binding_site],
                logp_values=cond_logp,
                descriptor_vectors=cond_descriptor,
                seed_smiles=[seed_smiles_value],
            )

            bos_id = self.smiles_tokenizer.bos_token_id
            eos_id = self.smiles_tokenizer.eos_token_id
            samples: List[str] = []

            for _ in range(num_samples):
                generated = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
                for _ in range(max_length - 1):
                    outputs = self.decoder(input_ids=generated, condition=condition)
                    logits = outputs["logits"][:, -1, :] / max(temperature, 1e-5)
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=-1)
                    if next_token.item() == eos_id:
                        break

                samples.append(
                    self.smiles_tokenizer.decode(
                        generated[0].tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )

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
    def __init__(
        self,
        model: ConditionalGNNSmilesGenerator,
        dataset: ConditionalHybridMoleculeDataset,
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

        max_active_tokens = int(attention_mask.sum(dim=1).max().item())
        max_active_tokens = max(max_active_tokens, 1)
        input_ids = input_ids[:, :max_active_tokens]
        attention_mask = attention_mask[:, :max_active_tokens]
        labels = labels[:, :max_active_tokens]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "protein": [item["protein"] for item in batch],
            "binding_site": [item["binding_site"] for item in batch],
            "seed_smiles": [item["seed_smiles"] for item in batch],
            "logp": torch.stack([item["logp"] for item in batch]),
            "descriptor_vector": torch.stack([item["descriptor_vector"] for item in batch]),
            "target_graph": [item["target_graph"] for item in batch],
            "ligand_graph": [item["ligand_graph"] for item in batch],
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
                batch["descriptor_vector"] = batch["descriptor_vector"].to(device)

                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - loss: {avg_loss:.4f}")