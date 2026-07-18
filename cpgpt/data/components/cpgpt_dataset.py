from pathlib import Path

import numpy as np
import sqlitedict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from cpgpt.log.utils import get_class_logger

from .dna_llm_embedder import DNALLMEmbedder


def _dependency_collection_for_species(species: str) -> str:
    """Map an Ensembl runtime species to a supported dependency collection."""
    return "human" if species in {"human", "homo_sapiens"} else "mammalian"


def _dependency_download_guidance(species: str) -> str:
    dependency_collection = _dependency_collection_for_species(species)
    return (
        "Download the dependencies first with Python:\n"
        "  from cpgpt import download_cpgpt\n"
        f"  resources = download_cpgpt(species='{dependency_collection}', "
        "force_download=True)"
    )


class CpGPTDataset(Dataset):
    """A custom PyTorch Dataset for handling CpG methylation data with DNA embeddings.

    This class processes and provides access to CpG methylation data along with
    corresponding DNA embeddings. It supports various data sorting strategies and
    integrates with different DNA language models.

    Attributes:
        processed_dir (str): Directory containing processed data files.
        max_length (int): Maximum number of CpG sites to include per sample.
        sorting_strategy (str): Strategy for sorting CpG sites.
        dna_context_len (int): Context length for DNA sequences.
        dna_llm (str): Name of the DNA language model to use.
        embedder (DNALLMEmbedder): Instance of DNALLMEmbedder for handling embeddings.
        logger: Logger instance for this class.
        dataset_metrics (Dict): Metrics for each dataset file.
        data_paths (List[str]): List of paths to dataset files.

    """

    def __init__(
        self,
        embedder: DNALLMEmbedder,
        processed_dir: str,
        max_length: int = 10000,
        sorting_strategy: str = "sorted_chromosome",
        dna_context_len: int = 2001,
        dna_llm: str = "nucleotide-transformer-v2-500m-multi-species",
        seed: int = 42,
    ) -> None:
        """Initialize the CpGData dataset.

        Args:
            embedder: An instance of DNALLMEmbedder.
            processed_dir: Directory containing processed data files.
            max_length: Maximum number of CpG sites to include per sample.
            sorting_strategy: Strategy for sorting CpG sites.
            dna_context_len: Context length for DNA sequences.
            dna_llm: Name of the DNA language model to use.
            seed: Seed for random number generation.

        Raises:
            ValueError: If any of the input parameters are invalid.

        """
        if not isinstance(embedder, DNALLMEmbedder):
            msg = f"embedder must be a DNALLMEmbedder, got {type(embedder)}."
            raise TypeError(msg)

        if not isinstance(processed_dir, str):
            msg = f"processed_dir must be strings, got {type(processed_dir)}."
            raise TypeError(msg)

        if not isinstance(max_length, int) or max_length <= 0:
            msg = f"max_length must be a positive integer, got {max_length}"
            raise TypeError(msg)

        if not isinstance(dna_context_len, int) or dna_context_len < 101:
            msg = f"dna_context_len must be an integer >= 101, got {dna_context_len}"
            raise TypeError(msg)

        valid_sorting_strategies = ["sorted_chromosome", "random_chromosome", "original", "random"]
        if sorting_strategy not in valid_sorting_strategies:
            msg = f"sorting_strategy must be one of {valid_sorting_strategies}"
            raise ValueError(msg)

        if dna_llm not in embedder.llm_embedding_size_dict:
            msg = f"dna_llm must be one of {embedder.llm_embedding_size_dict.keys()}"
            raise ValueError(msg)

        self.embedder = embedder
        self.processed_dir = processed_dir
        self.max_length = max_length
        self.sorting_strategy = sorting_strategy
        self.dna_context_len = dna_context_len
        self.dna_llm = dna_llm
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.logger = get_class_logger(self.__class__)

        self._load_dataset_metrics()

    def _load_dataset_metrics(self) -> None:
        """Load existing dataset metrics if available.

        Loads metrics from SQLite database containing information about processed datasets.
        Sets up data paths for accessing the dataset files.

        Raises:
            FileNotFoundError: If the dataset metrics file is not found

        """
        metrics_file = Path(self.processed_dir) / "dataset_metrics.db"
        if metrics_file.exists():
            with sqlitedict.SqliteDict(metrics_file, autocommit=False, outer_stack=False) as db:
                self.dataset_metrics = dict(db)
            self.data_paths = list(self.dataset_metrics.keys())
            self.logger.info("Loaded existing dataset metrics.")
        else:
            self.logger.error(
                "No existing dataset metrics found. Please run CpGPTDataSaver first.",
            )
            msg = "Dataset metrics file not found."
            raise FileNotFoundError(msg)

    def _apply_sorting_strategy(
        self,
        X: np.ndarray,
        var: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the specified sorting strategy to the data.

        Sorts CpG sites based on the chosen strategy:
        - sorted_chromosome: Sort by position within each chromosome
        - random_chromosome: Random order within each chromosome
        - random: Completely random order
        - original: Keep original order

        Args:
            X (np.ndarray): CpG methylation data array
            var (np.ndarray): Genomic variant information array

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - X: Sorted methylation data array
                - var: Sorted variant information array

        """
        if "chromosome" in self.sorting_strategy:
            chroms = var[:, 0]
            positions = var[:, 1]

            unique_chroms, chrom_indices = np.unique(chroms, return_inverse=True)

            sorted_indices = []
            for chrom in unique_chroms:
                chrom_mask = chrom_indices == np.where(unique_chroms == chrom)[0][0]
                chrom_positions = positions[chrom_mask]
                if "sorted" in self.sorting_strategy:
                    sorted_chrom_indices = np.argsort(chrom_positions)
                elif "random" in self.sorting_strategy:
                    sorted_chrom_indices = self.rng.permutation(len(chrom_positions))
                sorted_indices.extend(np.where(chrom_mask)[0][sorted_chrom_indices])

            # Shuffle chromosomes
            shuffled_chroms = self.rng.permutation(unique_chroms)

            # Apply the shuffle order to sorted indices
            final_sorted_indices = []
            for chrom in shuffled_chroms:
                chrom_mask = chroms == chrom
                chrom_sorted_indices = [i for i in sorted_indices if chrom_mask[i]]
                final_sorted_indices.extend(chrom_sorted_indices)

            X = X[final_sorted_indices]
            var = var[final_sorted_indices]

        elif self.sorting_strategy == "random":
            random_indices = self.rng.permutation(len(var))
            X = X[random_indices]
            var = var[random_indices]

        return X, var

    def _get_dna_embeddings(self, species: str, var: np.ndarray) -> np.ndarray:
        """Retrieve DNA embeddings based on genomic locations.

        Loads DNA embeddings from memory-mapped file for the given genomic positions.

        Args:
            species (str): Ensembl species name
            var (np.ndarray): Array containing chromosome and position information

        Returns:
            np.ndarray: Array of DNA embeddings for the given positions

        Raises:
            FileNotFoundError: If DNA embeddings file is missing
            KeyError: If species or genomic locations not found in embeddings

        """
        reverse_vocab_dict = self.embedder.ensembl_metadata_dict[species]["reverse_vocab"]
        embedding_size = self.embedder.llm_embedding_size_dict[self.dna_llm]
        embeddings_file = (
            Path(self.embedder.dependencies_dir)
            / "dna_embeddings"
            / species
            / self.dna_llm
            / f"{self.dna_context_len}bp_dna_embeddings.mmap"
        )

        # Check if the embeddings file exists
        if not embeddings_file.exists():
            # Check if the parent directories exist to provide more specific guidance
            dna_embeddings_dir = Path(self.embedder.dependencies_dir) / "dna_embeddings"
            species_dir = dna_embeddings_dir / species
            llm_dir = species_dir / self.dna_llm

            if not dna_embeddings_dir.exists():
                error_msg = (
                    f"DNA embeddings directory is missing: {dna_embeddings_dir}\n"
                    f"This suggests that dependencies for species '{species}' were not downloaded "
                    f"or the download was incomplete.\n\n"
                    f"{_dependency_download_guidance(species)}"
                )
            elif not species_dir.exists():
                error_msg = (
                    f"Species directory is missing: {species_dir}\n"
                    f"Dependencies for species '{species}' were not downloaded or are incomplete.\n\n"
                    f"{_dependency_download_guidance(species)}"
                )
            elif not llm_dir.exists():
                error_msg = (
                    f"DNA language model directory is missing: {llm_dir}\n"
                    f"Dependencies for species '{species}' and model '{self.dna_llm}' were not "
                    f"downloaded or are incomplete.\n\n"
                    f"{_dependency_download_guidance(species)}"
                )
            else:
                error_msg = (
                    f"DNA embeddings file is missing: {embeddings_file}\n"
                    f"This specific embeddings file for {self.dna_context_len}bp context length "
                    f"was not downloaded or is corrupted.\n\n"
                    f"{_dependency_download_guidance(species)}"
                )

            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        embeddings_index_dict = self.embedder.ensembl_metadata_dict[species][self.dna_llm][
            self.dna_context_len
        ]

        try:
            embeddings = np.memmap(
                embeddings_file,
                dtype="float32",
                mode="r",
                shape=(len(embeddings_index_dict), embedding_size),
            )
        except (OSError, ValueError) as e:
            error_msg = (
                f"Failed to open DNA embeddings file: {embeddings_file}\n"
                f"The file may be corrupted or incomplete. Original error: {e}\n\n"
                f"{_dependency_download_guidance(species)}"
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e

        try:
            embedding_indices = [
                embeddings_index_dict[f"{reverse_vocab_dict[chrom]}:{pos}"] for chrom, pos in var
            ]
        except KeyError as e:
            # Provide more helpful error for missing genomic locations
            missing_location = str(e).strip("'\"")
            error_msg = (
                f"Genomic location not found in embeddings index: {missing_location}\n"
                f"This could indicate a mismatch between the dataset and the available embeddings.\n"
                f"Make sure you're using the correct species ('{species}') and that the "
                f"embeddings were generated for the genomic coordinates in your dataset."
            )
            self.logger.error(error_msg)
            raise KeyError(error_msg) from e

        return embeddings[embedding_indices]

    def __len__(self) -> int:
        """Return the total number of samples across all files.

        Returns:
            int: Total number of samples in the dataset

        """
        return sum(self.dataset_metrics[path]["num_samples"] for path in self.data_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | None]:
        """Get a single item from the dataset.

        Retrieves methylation data, DNA embeddings, and metadata for the given index.
        Applies sorting strategy and length constraints if specified.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            Dict[str, Union[torch.Tensor, Optional[torch.Tensor]]]: Dictionary containing:
                - meth (torch.Tensor): CpG methylation data
                - dna_embeddings (torch.Tensor): DNA embeddings
                - chroms (torch.Tensor): Chromosome indices
                - positions (torch.Tensor): Genomic positions
                - obsm (Optional[torch.Tensor]): Additional observation matrix if available

        Raises:
            IndexError: If the index is out of range

        """
        for data_path in self.data_paths:
            dataset_name = (
                Path(data_path).with_suffix("").as_posix().replace("/", "_").replace("\\", "_")
            )
            dataset_dir = Path(self.processed_dir) / dataset_name

            # Load dataset shape
            X_shape_file = dataset_dir / "X_shape.npy"
            X_shape = np.load(X_shape_file, allow_pickle=True)
            n_obs, n_features = X_shape

            if idx < n_obs:
                # Load data files
                X_file = dataset_dir / "X.mmap"
                var_file = dataset_dir / "var.mmap"
                obsm_file = dataset_dir / "obsm.mmap"
                var_names_file = dataset_dir / "var_names.npy"
                species_file = dataset_dir / "species.npy"
                obsm_names_file = dataset_dir / "obsm_names.npy"

                # Load metadata
                var_names = np.load(var_names_file, allow_pickle=True)
                species = str(np.load(species_file, allow_pickle=True))
                obsm_names = (
                    np.load(obsm_names_file, allow_pickle=True)
                    if Path.exists(obsm_names_file)
                    else None
                )

                # Load data using memory mapping
                X = np.memmap(X_file, dtype="float32", mode="r", shape=(n_obs, n_features))
                var = np.memmap(
                    var_file,
                    dtype="int32",
                    mode="r",
                    shape=(n_features, len(var_names)),
                )
                obsm = (
                    np.memmap(obsm_file, dtype="float32", mode="r", shape=(n_obs, len(obsm_names)))
                    if Path.exists(obsm_file)
                    else None
                )

                # Apply max_length constraint if necessary
                if self.max_length < len(var):
                    if self.sorting_strategy != "original":
                        indices = self.rng.choice(
                            np.arange(len(var)),
                            size=min(self.max_length, len(var)),
                            replace=False,
                        )
                    else:
                        indices = np.arange(self.max_length)
                    X = X[idx, indices]
                    var = var[indices]
                else:
                    X = X[idx, :]
                    var = var[:]

                if obsm is not None:
                    obsm = obsm[idx, :]

                # Apply sorting strategy
                X, var = self._apply_sorting_strategy(X, var)

                # Get DNA embeddings
                embeddings = self._get_dna_embeddings(species, var)

                # Convert to PyTorch tensors
                X = torch.tensor(X, dtype=torch.float32)
                var = torch.tensor(var, dtype=torch.int32)
                embeddings = torch.tensor(embeddings, dtype=torch.float32)
                obsm = torch.tensor(obsm, dtype=torch.float32) if obsm is not None else None

                # Split var into chrom and pos
                chroms = var[:, 0]
                positions = var[:, 1]

                # Create output dictionary
                return {
                    "meth": X,
                    "dna_embeddings": embeddings,
                    "chroms": chroms,
                    "positions": positions,
                    "obsm": obsm,
                }

            idx -= n_obs

        self.logger.error("Index out of range.")
        msg = "Index out of range."
        raise IndexError(msg)


def cpgpt_data_collate(
    batch: list[dict[str, torch.Tensor | None]],
) -> dict[str, torch.Tensor | None]:
    """Custom collate function for batching CpGPT data.

    Pads sequences in the batch to the same length and combines them into batched tensors.

    Args:
        batch (List[Dict]): List of dictionaries from CpGPTDataset.__getitem__

    Returns:
        Dict[str, Union[torch.Tensor, Optional[torch.Tensor]]]: Dictionary containing:
            - meth (torch.Tensor): Batched methylation data [batch_size, max_len]
            - dna_embeddings (torch.Tensor): Batched DNA embeddings
                [batch_size, max_len, embed_dim]
            - chroms (torch.Tensor): Batched chromosome indices [batch_size, max_len]
            - positions (torch.Tensor): Batched positions [batch_size, max_len]
            - obsm (Optional[torch.Tensor]): Batched observation matrix if available

    Note:
        Uses padding_value=nan for methylation data and obsm
        Uses padding_value=0 for DNA embeddings
        Uses padding_value=-1 for chromosome and position indices

    """
    # Extract individual components from batch dictionaries
    meth = [item["meth"] for item in batch]
    dna_embeddings = [item["dna_embeddings"] for item in batch]
    chroms = [item["chroms"] for item in batch]
    positions = [item["positions"] for item in batch]
    obsm = [item["obsm"] for item in batch]

    # Create output dictionary with padded sequences
    output = {
        "meth": pad_sequence(meth, batch_first=True, padding_value=float("nan")),
        "dna_embeddings": pad_sequence(dna_embeddings, batch_first=True, padding_value=0),
        "chroms": pad_sequence(chroms, batch_first=True, padding_value=-1),
        "positions": pad_sequence(positions, batch_first=True, padding_value=-1),
    }

    # Handle obsm (which might be None for some samples)
    if all(o is not None for o in obsm):
        output["obsm"] = pad_sequence(obsm, batch_first=True, padding_value=float("nan"))
    else:
        output["obsm"] = None

    return output
