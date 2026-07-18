from pathlib import Path
from typing import Any

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .components.cpgpt_dataset import CpGPTDataset, cpgpt_data_collate
from .components.dna_llm_embedder import DNALLMEmbedder


class CpGPTDataModule(LightningDataModule):
    """LightningDataModule for CpGPT dataset.

    A PyTorch Lightning data module for handling CpGPT datasets, supporting training,
    validation, testing, and prediction workflows.

    """

    def __init__(
        self,
        train_dir: str | None = None,
        val_dir: str | None = None,
        test_dir: str | None = None,
        predict_dir: str | None = None,
        dependencies_dir: str | Path | None = None,
        batch_size: int = 4,
        num_workers: int = 4,
        max_length: int = 10000,
        dna_llm: str = "nucleotide-transformer-v2-500m-multi-species",
        dna_context_len: int = 2001,
        sorting_strategy: str = "random",
        pin_memory: bool = False,
        species: str = "human",
        hf_cache_dir: str | Path | None = None,
        hf_revision: str | None = None,
    ) -> None:
        """Initialize the CpGPT DataModule.

        Args:
            train_dir (Optional[str]): Directory containing training data. Defaults to None.
            val_dir (Optional[str]): Directory containing validation data. Defaults to None.
            test_dir (Optional[str]): Directory containing test data. Defaults to None.
            predict_dir (Optional[str]): Directory containing prediction data. Defaults to None.
            dependencies_dir: Explicit directory for model dependencies. When omitted,
                dependencies are resolved from the local Hugging Face cache.
            batch_size (int): Batch size for dataloaders. Defaults to 4.
            num_workers (int): Number of workers for data loading. Defaults to 4.
            max_length (int): Maximum sequence length. Defaults to 10000.
            dna_llm (str): DNA language model name. Defaults to
                "nucleotide-transformer-v2-500m-multi-species".
            dna_context_len (int): Context length for DNA sequences. Defaults to 2001.
            sorting_strategy (str): Strategy for sorting sequences. Defaults to "random".
            pin_memory (bool): Whether to pin memory in data loading. Defaults to False.
            species: Dependency collection to resolve. Defaults to "human".
            hf_cache_dir: Optional Hugging Face cache root.
            hf_revision: Optional Hugging Face repository revision.

        """
        super().__init__()

        # Check if at least one of the directories is provided
        if train_dir is None and val_dir is None and test_dir is None and predict_dir is None:
            msg = "At least one of the directories must be provided."
            raise ValueError(msg)

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        # Data paths
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.predict_dir = predict_dir

        # For DDP
        self.batch_size_per_device = batch_size

        # Data attributes
        self.data_train: CpGPTDataset | None = None
        self.data_val: CpGPTDataset | None = None
        self.data_test: CpGPTDataset | None = None
        self.data_predict: CpGPTDataset | None = None

        # Initialize embedder
        self.embedder = DNALLMEmbedder(
            dependencies_dir=dependencies_dir,
            species=species,
            cache_dir=hf_cache_dir,
            revision=hf_revision,
        )

    def prepare_data(self) -> None:
        """Prepare data for training.

        Note:
            This method is empty since data should be preprocessed beforehand.

        """

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage of training.

        Args:
            stage (Optional[str], optional): Stage of training ('fit', 'validate', 'test',
                or 'predict'). Defaults to None.

        Raises:
            RuntimeError: If batch size is not divisible by number of devices in
                distributed training.

        """
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                msg = (
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the "
                    f"number of devices ({self.trainer.world_size})."
                )
                raise RuntimeError(
                    msg,
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load datasets
        if self.hparams.train_dir is not None:
            self.data_train = CpGPTDataset(
                self.embedder,
                processed_dir=self.hparams.train_dir,
                max_length=self.hparams.max_length,
                sorting_strategy=self.hparams.sorting_strategy,
                dna_context_len=self.hparams.dna_context_len,
                dna_llm=self.hparams.dna_llm,
            )
        if self.hparams.val_dir is not None:
            self.data_val = CpGPTDataset(
                self.embedder,
                processed_dir=self.hparams.val_dir,
                max_length=10000,
                sorting_strategy=self.hparams.sorting_strategy,
                dna_context_len=self.hparams.dna_context_len,
                dna_llm=self.hparams.dna_llm,
            )
        if self.hparams.test_dir is not None:
            self.data_test = CpGPTDataset(
                self.embedder,
                processed_dir=self.hparams.test_dir,
                max_length=10000,
                sorting_strategy=self.hparams.sorting_strategy,
                dna_context_len=self.hparams.dna_context_len,
                dna_llm=self.hparams.dna_llm,
            )
        if self.hparams.predict_dir is not None:
            self.data_predict = CpGPTDataset(
                self.embedder,
                processed_dir=self.hparams.predict_dir,
                max_length=self.hparams.max_length,
                sorting_strategy=self.hparams.sorting_strategy,
                dna_context_len=self.hparams.dna_context_len,
                dna_llm=self.hparams.dna_llm,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the training dataloader.

        Returns:
            DataLoader[Any]: The training dataloader, or None if no training data is available.

        """
        if self.data_train is None:
            return None
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=cpgpt_data_collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            DataLoader[Any]: The validation dataloader, or None if no validation data is available.

        """
        if self.data_val is None:
            return None
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=cpgpt_data_collate,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            DataLoader[Any]: The test dataloader, or None if no test data is available.

        """
        if self.data_test is None:
            return None
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=cpgpt_data_collate,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the prediction dataloader.

        Returns:
            DataLoader[Any]: The prediction dataloader, or None if no prediction data is available.

        """
        if self.data_predict is None:
            return None
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=cpgpt_data_collate,
            drop_last=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Clean up resources after training.

        Args:
            stage (Optional[str], optional): Stage of training being torn down. Defaults to None.

        """

    def state_dict(self) -> dict[Any, Any]:
        """Get the state dictionary for checkpointing.

        Returns:
            Dict[Any, Any]: Empty dictionary as no state needs to be saved.

        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from a checkpoint.

        Args:
            state_dict (Dict[str, Any]): State dictionary from checkpoint.

        """


if __name__ == "__main__":
    _ = CpGPTDataModule()
