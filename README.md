# [ICLR 2025] Learning View-invariant World Models for Visual Robotic Manipulation

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

2. Running COMBO with the following command. A self-trained checkpoint can be found in "checkpoints/multiview_v0/model.pth" with the default model config in "configs/config.yaml". We provide three settings for evaluation:
* Training View: 
```bash
python rl_main.py --env_name ${your_metaworld_env_name} --env_mode "normal"
``` 
* Novel View(CIP): 
```bash
python rl_main.py --env_name ${your_metaworld_env_name} --env_mode "novel" --camera_change ${change_of_azimuth}
``` 
* Shaking View(CSH): 
```bash
python rl_main.py --env_name ${your_metaworld_env_name} --env_mode "shake"
``` 

## 😊 Acknowledgement
We would like to thank the authors of [OfflineRLKit](https://github.com/yihaosun1124/OfflineRL-Kit) for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation.

## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@inproceedings{pang2025reviwo,
  title={Learning View-invariant World Models for Visual Robotic Manipulation},
  author={Pang, jingcheng and Tang, nan and Li, kaiyuan, and Tang, Yuting and Cai, Xin-Qiang and Zhang, Zhen-Yu and Niu, Gang and Masashi, Sugiyama and Yu, yang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```