import os
import yaml

# Load config.yaml and process paths
def load_config():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as file:
        config = yaml.safe_load(file)

    # Expand environment variables and base path references
    base_path = os.path.expandvars(config["base_path"])
    config["video_dir"] = os.path.expandvars(config["video_dir"]).replace("${base_path}", base_path)
    config["frame_dir"] = os.path.expandvars(config["frame_dir"]).replace("${base_path}", base_path)

    return config

# Make config available when imported
config = load_config()
