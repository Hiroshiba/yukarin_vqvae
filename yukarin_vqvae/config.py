from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from yukarin_vqvae.utility import dataclass_utility
from yukarin_vqvae.utility.git_utility import get_branch_name, get_commit_id


class VocoderType(str, Enum):
    wavenet = "wavenet"
    wavernn = "wavernn"


@dataclass
class DatasetConfig:
    sampling_length: int
    dataset_glob: str
    num_train: Optional[int]
    num_test: int
    # evaluate_times: int
    # evaluate_time_second: float


@dataclass
class NetworkConfig:
    scaling_layer_num: int
    scaling_hidden_size: int
    residual_layer_num: int
    residual_hidden_size: int
    quantizer_embedding_num: int
    quantizer_embedding_size: int
    vocoder_type: VocoderType
    vocoder_hidden_size: int
    bin_size: int
    speaker_size: int
    speaker_embedding_size: int


@dataclass
class ModelConfig:
    quantize_loss_weight: float
    softmax_loss_weight: float


@dataclass
class TrainConfig:
    batchsize: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    quantizer_ema_decay: float
    num_processes: Optional[int] = None


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    pass
