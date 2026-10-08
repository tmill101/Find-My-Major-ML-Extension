from pathlib import Path

from src.config.config_loader import load_config
from src.export import export_frontend_artifacts


def main() -> None:
    """Command-line wrapper for exporting frontend inference artifacts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export frontend inference artifacts from experiment config."
    )
    parser.add_argument('config_path', help='Path to the experiment configuration YAML file')
    args = parser.parse_args()

    config = load_config(args.config_path)
    output_dir = Path(config['output']['directory']) / config['run']['run_id']
    output_dir.mkdir(parents=True, exist_ok=True)

    export_paths = export_frontend_artifacts(config, output_dir)
    print("Export complete.")
    print(f"ONNX model: {export_paths['onnx_model']}")
    print(f"Frontend database: {export_paths['frontend_db']}")


if __name__ == '__main__':
    main()
