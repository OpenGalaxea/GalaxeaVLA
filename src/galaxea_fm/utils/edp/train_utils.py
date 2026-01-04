import tarfile
import json
import os

from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from galaxea_fm.utils.edp.utils import get_training_record_meta

logger = get_logger(__name__)
EDP_STORE_DATASET_ROOT = "/edp-workspace/instance-env/datasets"


def extract_tar_gz(tar_path, extract_to=None):
    """
    Extract a .tar.gz file to the specified directory.
    
    Args:
        tar_path: Path to the .tar.gz file
        extract_to: Directory to extract to. If None, extracts to the same directory as tar_path
    
    Returns:
        Path to the extracted directory
    """
    tar_path = Path(tar_path)
    
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")
    
    if extract_to is None:
        extract_to = tar_path.parent
    else:
        extract_to = Path(extract_to)
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {tar_path} to {extract_to}")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        logger.info(f"Extraction complete: {tar_path} -> {extract_to}")
    except Exception as e:
        logger.error(f"Failed to extract {tar_path}: {e}")
        raise
    
    return extract_to


def override_cfg_with_edp(cfg: DictConfig, output_dir: Path):
    is_use_edp = os.environ.get("TRAINING_RECORD_NAME", None) is not None
    if is_use_edp:
        logger.info(f"Using EDP for training, training record name: {os.environ.get('TRAINING_RECORD_NAME')}")
        training_record_meta = get_training_record_meta(os.environ.get("TRAINING_RECORD_NAME"))
        logger.info(f"Get training record meta from EDP: {json.dumps(training_record_meta, indent=2)}")

        if training_record_meta.get("data", None) is not None:
            # NOTE: Set dataset path in edp container
            edp_dataset_roots = training_record_meta["data"]["trainingDataSetDirList"]
            # NOTE: the dataset's tar file is stored in the EDP container, so we need to extract it
            # edp_dataset_root: <EDP_path_in_container>/<dataset_name_in_edp>/<dataset_version_in_edp>
            # dataset_name: <dataset_name_in_edp>
            # Tar path format: <edp_dataset_root>/<dataset_name>/<dataset_name>.tar.gz
            dataset_names = []
            real_dataset_roots = []
            for edp_root in edp_dataset_roots:
                name = edp_root.split("/")[-2]
                real_dataset_root = os.path.join(edp_root, name)
                # stored large file in edp dataset (e.g. pretrain model, etc.)
                if "test-rawdata" in edp_root:
                    os.environ["HF_HUB_CACHE"] = os.path.join(edp_root, "hub_cache")
                    logger.info(f"Set HF_HUB_CACHE to {os.environ['HF_HUB_CACHE']}")
                    # TODO: support to load g0 or dp pretrained model
                else:
                    dataset_names.append(name)
                    real_dataset_roots.append(real_dataset_root)

            tmp_extract_paths = []
            for root, name in zip(real_dataset_roots, dataset_names):
                dataset_tar_path = os.path.join(root, f"{name}.tar.gz")
                # Extract the tar.gz file to <edp_store_root> (for read-only filesystem containers)
                extract_tar_gz(dataset_tar_path, extract_to=EDP_STORE_DATASET_ROOT)

                tmp_extract_paths.append(os.path.join(EDP_STORE_DATASET_ROOT, name))

            cfg.data.dataset_dirs = tmp_extract_paths
            # NOTE: training task name is same as the model card name
            cfg.edp.card = training_record_meta["data"]["trainingRecordName"]
            
            # NOTE: Override training config from training_record_meta (e.g., max_steps, batch_size, etc.)
            edp_config = None
            if training_record_meta["data"].get("config", None) is not None:
                edp_config = training_record_meta["data"]["config"]
                
                # Check if keys exist using OmegaConf.select()
                def get_all_keys(d, parent_key=''):
                    """Flatten nested dict keys to dot notation"""
                    keys = []
                    for k, v in d.items():
                        full_key = f"{parent_key}.{k}" if parent_key else k
                        keys.append(full_key)
                        if isinstance(v, dict):
                            keys.extend(get_all_keys(v, full_key))
                    return keys
                
                missing_keys = [k for k in get_all_keys(edp_config) if OmegaConf.select(cfg, k) is None]
                if missing_keys:
                    logger.warning(f"The following keys from EDP config do not exist in cfg: {missing_keys}")
                
                # Update cfg with edp_config
                cfg = OmegaConf.merge(cfg, edp_config)
                logger.info(f"Updated cfg with EDP config: {json.dumps(OmegaConf.to_container(edp_config), indent=2)}")
            
            # NOTE: Resumed training from model in EDP
            if training_record_meta["data"].get("modelDir", None) is not None:
                if training_record_meta["data"]["modelDir"].endswith(".pt"):
                    cfg.resume_ckpt = training_record_meta["data"]["modelDir"]
                    logger.info(f"Resuming training from model in EDP: {training_record_meta['data']['modelDir']}")
                else:
                    raise ValueError(f"Model in EDP is not a .pt file: {training_record_meta['data']['modelDir']}")
        else:
            raise ValueError(f"Training record meta does not contain data: {training_record_meta}")

        # Save updated config to Hydra output directory (after all runtime modifications)
        hydra_cfg_dir = output_dir / ".hydra"
        if hydra_cfg_dir.exists():
            # Save the updated full config
            with open(hydra_cfg_dir / "config.yaml", "w") as f:
                OmegaConf.save(config=cfg, f=f)
            logger.info(f"Updated Hydra config saved to {hydra_cfg_dir / 'config.yaml'}")
            
            # Optionally, update overrides.yaml with runtime modifications
            # Read existing overrides
            overrides_path = hydra_cfg_dir / "overrides.yaml"
            if overrides_path.exists():
                with open(overrides_path, "r") as f:
                    existing_overrides = f.read().strip().split('\n')
            else:
                existing_overrides = []
            
            # Add runtime modifications as overrides (if using EDP)
            def flatten_config(d, parent_key=''):
                """Flatten nested dict to dot notation key-value pairs"""
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_config(v, new_key))
                    else:
                        items.append(f"{new_key}={v}")
                return items
            
            runtime_overrides = [
                f"data.dataset_dirs={cfg.data.dataset_dirs}",
                f"edp.card={cfg.edp.card}",
                f"resume_ckpt={cfg.resume_ckpt}",
            ]
            if edp_config is not None:
                runtime_overrides.extend(flatten_config(OmegaConf.to_container(edp_config)))
            # Append runtime overrides with a comment
            with open(overrides_path, "w") as f:
                for override in existing_overrides:
                    f.write(f"{override}\n")
                f.write("# Runtime modifications:\n")
                for override in runtime_overrides:
                    f.write(f"- {override}\n")
            logger.info(f"Updated overrides saved to {overrides_path}")
        else:
            raise ValueError(f"Could not find hydra config in {hydra_cfg_dir}")
    else:
        logger.info("Not using EDP, using local data")
    
    return is_use_edp
