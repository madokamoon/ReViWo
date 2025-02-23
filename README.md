# [ICLR 2025] Reviwo

## 🔧 Python Environment Configuration
1. Update the `prefix` parameter in `environment.yml`
2. Build Python environment with following command
```bash
conda env create -f environment.yml
```

## 🚀 View-invariant Encoder Training
1. Collect the multi-view data from Metaworld with the following command, make sure you have installed mujoco, and we recommend using mujoco-210.
```bash
python collect_data/collect_multi_view_data.py
```

2. Train the view-invariant encoder by running, the configs of training is referred to path`configs/config.yaml`:
```bash
python tokenizer_main.py
```

## 🦾 Running COMBO with the learnt view-invariant encoder
1. Collect the single-view data for COMBO with the following command:
```bash
python collect_data/collect_world_model_training_data.py --env_name ${your_metaworld_env_name}
```

2. Running COMBO with the following command. We provide three setting for evaluation:
* Training View: 
```bash
python rl_main.py --env_name ${your_metaworld_env_name} --env_mode "normal"
``` 
* Novel View(CIP): 
```bash
python rl_main.py --env_name ${your_metaworld_env_name} --env_mode "novel" --azimuth ${change_of_azimuth}
``` 
* Shaking View(CSH): 
```bash
python rl_main.py --env_name ${your_metaworld_env_name} --env_mode "novel" --azimuth ${change_of_azimuth}
``` 

## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@inproceedings{pang2025reviwo,
  title={Learning View-invariant World Models for Visual Robotic Manipulation},
  author={Pang, jingcheng and Tang, nan and Li, kaiyuan, and Yu, yang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```