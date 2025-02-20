from .multiview_tokenizer import (
        project_dir, 
        MultiViewTokenizer
    )   

def init_multiview_tokenizer(load_ckpt_path: str = "/checkpoints/multiview_v0/model.pth"):
    import ruamel.yaml as yaml
    from pathlib import Path
    import common

    configs = yaml.safe_load(
        (Path(project_dir + "/configs") / "config.yaml").read_text()
    )
    config_name = "defaults"

    parsed, remaining = common.Flags(configs=[config_name]).parse(known_only=True)
    config = common.Config(configs[config_name])
    
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining, known_only=True)[0]

    if not project_dir in load_ckpt_path:
        load_ckpt_path = project_dir + load_ckpt_path
    config = config.update({
        "load_ckpt_path": load_ckpt_path
    })

    tokenizer = MultiViewTokenizer(config)
    return tokenizer