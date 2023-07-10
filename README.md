# Med-Flamingo  

<img src="img/logo.png" width="100">

This is the code repo for the Med-Flamingo paper.

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


