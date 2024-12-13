<p align="center">
 <!-- <h2 align="center">👗 HieraFashDiff: Hierarchical Fashion Design with Multi-stage Diffusion Models</h2> -->
 <h2 align="center">👗 HieraFashDiff: Hierarchical Fashion Design with Multi-stage Diffusion Models</h2>
 <p align="center"> 
    Zhifeng Xie · Hao Li · Huiming Ding · Mengtian Li · Xinhan Di · Ying Cao<sup>*</sup>
 </p>
 <p align="center"> 
    <b>AAAI 2025</b>
 </p>
  <p align="center"> <sup>*</sup> <i>Corresponding author</i> </p>
</p>

 </p>

[![Website](assets/badge-website.svg)](https://haoli-zbdbc.github.io/hierafashdiff.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2401.07450)


![Cover](/assets/teaser_w.png)


## 📻 Overview
<div align="justify">
Fashion design is a challenging and complex process.
Recent works on fashion generation and editing are all agnostic of the actual fashion design process, which limits their usage in practice.
In this paper, we propose a novel hierarchical diffusion-based framework tailored for fashion design, coined as <b>HieraFashDiff</b>. 
Our model is designed to mimic the practical fashion design workflow, by unraveling the denosing process into two successive stages: 1) an ideation stage that generates design proposals given high-level concepts and 2) an iteration stage that continuously refines the proposals using low-level attributes. 
Our model supports fashion design generation and fine-grained local editing in a single framework. 
To train our model, we contribute a new dataset of full-body fashion images annotated with hierarchical text descriptions. 
Extensive evaluations show that, as compared to prior approaches, our method can generate fashion designs and edited results with higher fidelity and better prompt adherence, showing its promising potential to augment the practical fashion design workflow. 
</div>

## 🛠️ Setup

This setup was tested with `Ubuntu 22.04.4 LTS`, `CUDA Version: 11.3`, and `Python 3.8.5`.

First, clone the github repo...

```bash
git clone git@github.com:haoli-zbdbc/hierafashdiff.git
cd hierafashdiff
```

Then download the weights via

```bash
wget https://huggingface.co/vmip-shu/HieraFashDiff/resolve/main/hfd_100epochs.ckpt?download=true -P checkpoints/
```

Now you have either the option to setup a virtual environment and install all required packages with `pip`

```bash
pip install -r requirements.txt
```

or if you prefer to use `conda` create the conda environment via

```bash
conda env create -f environment.yml
```

or `docker` deploy

```shell
docker build -t hfddm .
docker run -it -d --restart always --shm-size 128g --gpus device=0 --name hfddm -v /path/to/HieraFashDiff:/app -p 0.0.0.0:8000:8000 hfddm
```


Now you should be able to design! 👗


## 🚀 Usage

You can just run the python script `gradio_hfd.py` as follows

```bash
python gradio_hfd.py
```


## Data Preparation
You need to download HieraFashion dataset from [Google Drive](https://drive.google.com/drive/folders/1WDo6DaNl6mMgCXuxmnGxNHTlsNSryoJA?usp=sharing) and unzip to your own path `/path/to/HieraFashion`. The dataset folder structure should be as follows:
```
HieraFashion
├── train_images
│   ├── 000000_0.jpg
│   ├── .......
│   └── WOMEN-Sweatshirts_Hoodies-id_00006976-01_4_full.jpg
├── test_images
│   ├── 78_0.jpg
│   ├── .......
│   └── WOMEN-Sweatshirts_Hoodies-id_00007240-01_4_full.jpg
├── train_pose
│   ├── json
│   └── pose_images
├── test.json
├── train.json
└── train_pose.json

```

## Training 

> [!IMPORTANT]
> Replace all the `/path/to` paths in the code and configuration files with real paths.
> `/path/to` paths exist in all the configuration files under the folder `utils/config.py` and run `huggingface-cli download --resume-download laion/CLIP-ViT-H-14-laion2B-s32B-b79K --local-dir checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K`.


### Load the dataset
You need to write a simple script to read this dataset for pytorch. (In fact we have written it for you in `scripts/train/my_dataset.py`.)

```bash
python scripts/train/my_dataset.py
```

### Train
Train the code with the following command:

```shell
python scripts/train/train_cldm.py
```

Note that we first fine-tune the stable diffusion model on the [Dress Code Multimodal](https://github.com/aimagelab/multimodal-garment-designer).You can download the pre-trained model(control_dresscode_ini.ckpt) from [Huggingface](https://huggingface.co/vmip-shu/HieraFashDiff/tree/main).


## Trend

[![Star History Chart](https://api.star-history.com/svg?repos=haoli-zbdbc/hierafashdiff&type=Date)](https://star-history.com/#haoli-zbdbc/hierafashdiff&Date)




## 🎓 Citation

Please cite our paper:

```bibtex
   @article{xie2024hierarchical,
      title={HieraFashDiff: Hierarchical Fashion Design with Multi-stage Diffusion Models},
      author={Xie, Zhifeng and Li, Hao and Ding, Huiming and Li, Mengtian and Di, Xinhan and Cao, Ying},
      journal={arXiv preprint arXiv:2401.07450},
      year={2024}
   }
```






