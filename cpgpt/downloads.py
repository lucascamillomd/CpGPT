"""Download and resolve CpGPT resources hosted on Hugging Face."""

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

MODEL_REPO_ID = "lucascamillomd/cpgpt-models"
DEPENDENCY_REPO_IDS = {
    "human": "lucascamillomd/cpgpt-human-dependencies",
    "mammalian": "lucascamillomd/cpgpt-mammalian-dependencies",
}


@dataclass(frozen=True)
class DownloadedCpGPTResources:
    """Paths for CpGPT resources downloaded from Hugging Face."""

    model_name: str | None = None
    species: str | None = None
    checkpoint_path: Path | None = None
    config_path: Path | None = None
    vocab_path: Path | None = None
    dependencies_path: Path | None = None


def _validate_species(species: str) -> None:
    if species not in DEPENDENCY_REPO_IDS:
        supported = ", ".join(DEPENDENCY_REPO_IDS)
        raise ValueError(f"Unsupported species '{species}'. Choose from: {supported}.")


def _download_model_file(
    filename: str,
    *,
    cache_dir: str | Path | None,
    local_dir: Path | None,
    revision: str | None,
    force_download: bool,
    token: str | bool | None,
    local_files_only: bool,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=filename,
            cache_dir=cache_dir,
            local_dir=local_dir,
            revision=revision,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
    )


def _model_not_found_message(model: str, *, cached: bool) -> str:
    location = (
        "selected Hugging Face cache" if cached else "CpGPT Hugging Face repository"
    )
    return (
        f"CpGPT model '{model}' is not available in the {location}. "
        "Download it first with Python:\n"
        "  from cpgpt import download_cpgpt\n"
        f"  resources = download_cpgpt(model='{model}')"
    )


def _dependencies_not_found_message(species: str) -> str:
    return (
        f"CpGPT dependencies for '{species}' are not available from Hugging Face "
        "or in the selected cache. Download them first with Python:\n"
        "  from cpgpt import download_cpgpt\n"
        f"  resources = download_cpgpt(species='{species}')"
    )


def _resolve_model(
    model: str,
    *,
    cache_dir: str | Path | None,
    local_dir: Path | None,
    revision: str | None,
    force_download: bool,
    token: str | bool | None,
    local_files_only: bool,
) -> DownloadedCpGPTResources:
    try:
        checkpoint_path = _download_model_file(
            f"weights/{model}.ckpt",
            cache_dir=cache_dir,
            local_dir=local_dir,
            revision=revision,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
        config_path = _download_model_file(
            f"config/{model}.yaml",
            cache_dir=cache_dir,
            local_dir=local_dir,
            revision=revision,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
    except LocalEntryNotFoundError as error:
        if not local_files_only:
            raise
        raise FileNotFoundError(
            _model_not_found_message(model, cached=True)
        ) from error
    except EntryNotFoundError as error:
        raise FileNotFoundError(
            _model_not_found_message(model, cached=local_files_only)
        ) from error

    try:
        vocab_path = _download_model_file(
            f"vocab/{model}.json",
            cache_dir=cache_dir,
            local_dir=local_dir,
            revision=revision,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
    except LocalEntryNotFoundError:
        if not local_files_only:
            raise
        vocab_path = None
    except EntryNotFoundError:
        vocab_path = None

    return DownloadedCpGPTResources(
        model_name=model,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        vocab_path=vocab_path,
    )


def _resolve_dependencies(
    species: str,
    *,
    cache_dir: str | Path | None,
    local_dir: Path | None,
    revision: str | None,
    force_download: bool,
    token: str | bool | None,
    local_files_only: bool,
) -> Path:
    try:
        return Path(
            snapshot_download(
                repo_id=DEPENDENCY_REPO_IDS[species],
                cache_dir=cache_dir,
                local_dir=local_dir,
                revision=revision,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )
        )
    except LocalEntryNotFoundError as error:
        if not local_files_only:
            raise
        raise FileNotFoundError(_dependencies_not_found_message(species)) from error
    except EntryNotFoundError as error:
        raise FileNotFoundError(_dependencies_not_found_message(species)) from error


def download_cpgpt(
    *,
    model: str | None = None,
    species: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    token: str | bool | None = None,
) -> DownloadedCpGPTResources:
    """Download selected CpGPT resources from their public Hugging Face repos.

    Args:
        model: Model name used for its checkpoint, configuration, and optional
            vocabulary filenames.
        species: Dependency collection to download; either ``"human"`` or
            ``"mammalian"``.
        cache_dir: Optional override for the native Hugging Face cache location.
        local_dir: Optional materialization root. Model files use
            ``<local_dir>/model`` and dependencies use ``<local_dir>/<species>``;
            when omitted, downloads remain in the native Hugging Face cache.
        revision: Optional Hub branch, tag, or commit.
        force_download: Whether to download files again when they are cached.
        token: Optional Hugging Face access token or token-discovery flag.

    Returns:
        DownloadedCpGPTResources: Paths for each requested resource.

    Raises:
        ValueError: If neither resource type is requested or species is unsupported.
        FileNotFoundError: If a required resource is missing remotely or from a
            cache-only lookup. Hub authentication and network errors propagate.
    """
    if model is None and species is None:
        raise ValueError("At least one of model or species must be provided.")
    if species is not None:
        _validate_species(species)

    root = Path(local_dir) if local_dir is not None else None
    model_resources = DownloadedCpGPTResources()
    if model is not None:
        model_resources = _resolve_model(
            model,
            cache_dir=cache_dir,
            local_dir=root / "model" if root is not None else None,
            revision=revision,
            force_download=force_download,
            token=token,
            local_files_only=False,
        )

    dependencies_path = None
    if species is not None:
        dependencies_path = _resolve_dependencies(
            species,
            cache_dir=cache_dir,
            local_dir=root / species if root is not None else None,
            revision=revision,
            force_download=force_download,
            token=token,
            local_files_only=False,
        )

    return DownloadedCpGPTResources(
        model_name=model_resources.model_name,
        species=species,
        checkpoint_path=model_resources.checkpoint_path,
        config_path=model_resources.config_path,
        vocab_path=model_resources.vocab_path,
        dependencies_path=dependencies_path,
    )


def resolve_cached_model(
    model: str,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> DownloadedCpGPTResources:
    """Resolve a model from the local Hugging Face cache without network access."""
    return _resolve_model(
        model,
        cache_dir=cache_dir,
        local_dir=None,
        revision=revision,
        force_download=False,
        token=None,
        local_files_only=True,
    )


def resolve_cached_dependencies(
    species: str,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> Path:
    """Resolve species dependencies from the cache without network access."""
    _validate_species(species)
    return _resolve_dependencies(
        species,
        cache_dir=cache_dir,
        local_dir=None,
        revision=revision,
        force_download=False,
        token=None,
        local_files_only=True,
    )
