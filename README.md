

![demo](https://github.com/user-attachments/assets/d91a9389-205b-4e51-959f-b0a069131fef)


# NZ-A

[![Python Version](https://img.shields.io/badge/python-3.12-orange)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-blue)](https://opensource.org/licenses/Apache-2.0)
![Paper](https://img.shields.io/badge/Paper-green)
![Demo](https://img.shields.io/badge/Demo-red)
![Demo](https://img.shields.io/badge/Updating-blue)

This repo provides code for NanoZyme-Agent (NZ-A), a framework equipped with LLM-based agents and customized AI toolkits for material research. 

Using NZ-A, researchers successfully identify six unreported rare earth-based POD-like nanozymes from nearly 600k candidate materials within seconds. 

**Note: The current repo ONLY represents a minimal implementation and will be iteratively refined throughout the peer-review process.**

## :zap: Online Demo

We are developing an ​​online demo​​ for researchers ​​without any programming expertise​​. The current interface is shown below, and the ​​web service​​ will be released during the peer-review process.

https://github.com/user-attachments/assets/4f8671ae-79b1-480a-ac6f-88248a464970

## :package: Installation

NZ-A is intended to work with Python 3.12. Installation can be done via pip:

```
pip install -r requirements.txt
```

## :books: Dataset

**Train dataset**

This study incoprates [Wei dataset](http://nanozymes.net), [Dizyme dataset](https://dizyme.aicidlab.itmo.ru/), and [Huang dataset](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202201736) for model training. To avoide compyright issues, researchers should download these datasets according to the prodived url and organize the dataset within the `./data` directory, with files aligned with the format of `./data/example_train.csv`.

**Screen dataset**

The trained model is used to screen the nanozymes in [Materials Project](https://next-gen.materialsproject.org/), [Aflow](https://aflowlib.org/), and [OQMD](https://oqmd.org/). Researchers should download these datasets via the provided API and store them in the './data' directory, ensuring the internal data structure aligns with the format of `./data/example_database.csv`.

## :rocket: Launching NZ-A
Researchers should launch NZ-A trhough the folloing command:
```
python app.py
```
Then, NZ-A can be accessed in http://127.0.0.1:5000/ via web browser.

After entering [DeepSeek API key](https://api-docs.deepseek.com/), researchers can perform nanozymes screening through natural language :grin:

## :scroll: Citation
We will update this block after the corresponding manuscript acceptance. 

## :bookmark: License
This project is licensed under the Apache-2.0 License.

## :pray: Acknowledgement

We gratefully acknowledge the following studies which provides valuable dataset for us.

```
@article{li2022data,
  title={Data-informed discovery of hydrolytic nanozymes},
  author={Li, Sirong and Zhou, Zijun and Tie, Zuoxiu and Wang, Bing and Ye, Meng and Du, Lei and Cui, Ran and Liu, Wei and Wan, Cuihong and Liu, Quanyi and others},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={827},
  year={2022}
}

@article{wu2019nanomaterials,
  title={Nanomaterials with enzyme-like characteristics (nanozymes): next-generation artificial enzymes (II)},
  author={Wu, Jiangjiexing and Wang, Xiaoyu and Wang, Quan and Lou, Zhangping and Li, Sirong and Zhu, Yunyao and Qin, Li and Wei, Hui},
  journal={Chemical Society Reviews},
  volume={48},
  number={4},
  pages={1004--1076},
  year={2019}
}

@article{razlivina2022dizyme,
  title={DiZyme: open-access expandable resource for quantitative prediction of nanozyme catalytic activity},
  author={Razlivina, Julia and Serov, Nikita and Shapovalova, Olga and Vinogradov, Vladimir},
  journal={Small},
  volume={18},
  number={12},
  pages={2105673},
  year={2022}
}

@article{razlivina2024ai,
  title={AI-Powered knowledge base enables transparent prediction of nanozyme multiple catalytic activity},
  author={Razlivina, Julia and Dmitrenko, Andrei and Vinogradov, Vladimir},
  journal={The Journal of Physical Chemistry Letters},
  volume={15},
  number={22},
  pages={5804--5813},
  year={2024}
}

@article{wei2022prediction,
  title={Prediction and design of nanozymes using explainable machine learning},
  author={Wei, Yonghua and Wu, Jin and Wu, Yixuan and Liu, Hongjiang and Meng, Fanqiang and Liu, Qiqi and Midgley, Adam C and Zhang, Xiangyun and Qi, Tianyi and Kang, Helong and others},
  journal={Advanced materials},
  volume={34},
  number={27},
  pages={2201736},
  year={2022}
}

@article{jain2013commentary,
  title={Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},
  author={Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and others},
  journal={APL materials},
  volume={1},
  number={1},
  year={2013}
}

@article{curtarolo2012aflow,
  title={AFLOW: An automatic framework for high-throughput materials discovery},
  author={Curtarolo, Stefano and Setyawan, Wahyu and Hart, Gus LW and Jahnatek, Michal and Chepulskii, Roman V and Taylor, Richard H and Wang, Shidong and Xue, Junkai and Yang, Kesong and Levy, Ohad and others},
  journal={Computational Materials Science},
  volume={58},
  pages={218--226},
  year={2012}
}

@article{saal2013materials,
  title={Materials design and discovery with high-throughput density functional theory: the open quantum materials database (OQMD)},
  author={Saal, James E and Kirklin, Scott and Aykol, Muratahan and Meredig, Bryce and Wolverton, Christopher},
  journal={Jom},
  volume={65},
  pages={1501--1509},
  year={2013}
}

@article{kirklin2015open,
  title={The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies},
  author={Kirklin, Scott and Saal, James E and Meredig, Bryce and Thompson, Alex and Doak, Jeff W and Aykol, Muratahan and R{\"u}hl, Stephan and Wolverton, Chris},
  journal={npj Computational Materials},
  volume={1},
  number={1},
  pages={1--15},
  year={2015}
}
```
