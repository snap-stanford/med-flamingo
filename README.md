# Med-Flamingo  

<img src="img/logo.png" width="100">

This is the code repo for the [Med-Flamingo paper](https://arxiv.org/abs/2307.15189).

More updates to follow soon! 



## Setup  

Create virtual environment, e.g.:  

```$ virtualenv flam_env```  
```$ source flam_env/bin/activate```  

Install dependencies: (we assume GPU device / cuda available) 

```$ source install.sh```  


### Setting up Llama-7B (v1) locally  

Due to some recent changes in tokenizer class names, directly using the hf space may lead to problems.  
We recommend to manually download the model, e.g. in a new dir models/ the following way:  

```$ git lfs install```  
```$ git clone https://huggingface.co/decapoda-research/llama-7b-hf```  

In tokenizer_config.json, set:  
```"tokenizer_class": "LlamaTokenizer"```  
Now, you should be all set.

## Demo  

1. Go to scripts/   

2. Edit demo.py and enter your Llama-7B path (v1).  

3. Run:

```$python demo.py``` 



## Citing  


If you found this repository interesting, please consider citing our pre-print:  

```bibtex
@article{moor2023medflamingo,
    title={Med-Flamingo: A Multimodal Medical Few-shot Learner},
    author={Moor, Michael and Huang, Qian and Wu, Shirley and Yasunaga, Michihiro and Zakka, Cyril and Dalmia, Yash and Reis, Eduardo Pontes and Rajpurkar, Pranav and Leskovec, Jure},
    year={2023},
    month={July},
    note={arXiv:2307.15189},
    url={https://arxiv.org/abs/2307.15189}
}  
```

Furthermore, the following two references enabled our project in the first place:  

```bibtex
@software{anas_awadalla_2023_7733589,
  author = {Awadalla, Anas and Gao, Irena and Gardner, Joshua and Hessel, Jack and Hanafy, Yusuf and Zhu, Wanrong and Marathe, Kalyani and Bitton, Yonatan and Gadre, Samir and Jitsev, Jenia and Kornblith, Simon and Koh, Pang Wei and Ilharco, Gabriel and Wortsman, Mitchell and Schmidt, Ludwig},
  title = {OpenFlamingo},
  month        = mar,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.1.1},
  doi          = {10.5281/zenodo.7733589},
  url          = {https://doi.org/10.5281/zenodo.7733589}
}
```

```bibtex
@article{Alayrac2022FlamingoAV,
  title={Flamingo: a Visual Language Model for Few-Shot Learning},
  author={Jean-Baptiste Alayrac and Jeff Donahue and Pauline Luc and Antoine Miech and Iain Barr and Yana Hasson and Karel Lenc and Arthur Mensch and Katie Millican and Malcolm Reynolds and Roman Ring and Eliza Rutherford and Serkan Cabi and Tengda Han and Zhitao Gong and Sina Samangooei and Marianne Monteiro and Jacob Menick and Sebastian Borgeaud and Andy Brock and Aida Nematzadeh and Sahand Sharifzadeh and Mikolaj Binkowski and Ricardo Barreira and Oriol Vinyals and Andrew Zisserman and Karen Simonyan},
  journal={ArXiv},
  year={2022},
  volume={abs/2204.14198}
}
```
