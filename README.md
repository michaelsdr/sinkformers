# Code for the paper : "_Sinkformers:  Transformers with Doubly Stochastic Attention_"

## Paper
You will find our paper [here](https://arxiv.org/abs/2110.11773).

## Compat

This package has been developed and tested with `python3.8`. It is therefore not guaranteed to work with earlier versions of python.

## Install the repository on your machine


This package can easily be installed using `pip`, with the following command:

```bash
pip install numpy
pip install -e .
```

This will install the package and all its dependencies, listed in `requirements.txt`.

**Each command has to be executed from the root folder** sinkformers. 
Our code is distributed in the different repositories. For each repository, we modify the architectures proposed by replacing the _SoftMax_ attention with a _Sinkhorn_ attention.  

## Defining a toy Sinkformer for which attention matrices are doubly stochastic

For this example we use a Transformer from the nlp-tutorial library and define its Sinkformer counterpart with the argument "n_it", the number of iterations in Sinkhorn's algorithm. 

```bash
cd nlp-tutorial/text-classification-transformer
```

```python
import torch
from model import TransformerEncoder
n_it = 1
print('1 iteration in Sinkhorn corresponds to the original Transformer: ')
transformer = TransformerEncoder(vocab_size=1000, seq_len=512, n_layers=1,  n_heads=1, n_it=n_it, print_attention=True, pad_id=-1)
inp = torch.arange(512).repeat(5, 1)
out = transformer(inp)
n_it = 5
print('5 iteration in Sinkhorn gives a Sinkformer with perfectly doubly stochastic attention matrices: ')
sinkformer = TransformerEncoder(vocab_size=1000, seq_len=512, n_layers=1,  n_heads=1, n_it=n_it, print_attention=True, pad_id=-1)
inp = torch.arange(512).repeat(5, 1)
out = sinkformer(inp)
```


Then go back to the root:

```bash
cd ..
cd ..
```

## Reproducing the experiments of the paper

**Comparison of the different normalizations.**

```bash
python plot_normalizations.py
```

**ModelNet 40 classification.** Code adapted from this [repository](https://github.com/juho-lee/set_transformer).
First, you need to preprocess the ModelNet40 dataset available [here](http://modelnet.cs.princeton.edu/). Unzip it and save it under model_net_40/data. Then, preferably on multiple cpus, run

```bash
cd model_net_40
python to_h5.py
python formatting.py
cd ..
mv model_net_40/data/ModelNet40_cloud.h5 set_transformer/ModelNet40_cloud.h5
cd set_transformer
mkdir ../dataset
mv ModelNet40_cloud.h5 ../dataset/ModelNet40_cloud.h5
cd ..
```

Then you can train a Set Sinkformer (or Set Transformer) on ModelNet 40 with 

```bash
cd set_transformer
python one_expe.py
cd ..
```

Arguments for one_expe.py can be accessed through 

```bash
cd set_transformer
python one_expe.py --help
cd ..
```

Results are saved in the folder set_transformer/results. You can plot the learning curves using the script set_transformer/plot_results.py. The array _iterations_ in the script must contains the different values for n_it used when training. 


**Sentiment Analysis.** Code adapted from this [repository](https://github.com/lyeoni/nlp-tutorial/tree/master/text-classification-transformer). You can also train a Sinkformer for Sentiment Analysis on the IMDb Dataset with the following command (the IMDb Dataset is downloaded automatically).

```bash
cd nlp-tutorial/text-classification-transformer
python one_expe.py
cd ..
cd ..
```

Arguments for one_expe.py can be accessed through 

```bash
cd nlp-tutorial/text-classification-transformer
python one_expe.py --help
cd ..
```

Results are saved in the folder nlp-tutorial/text-classification-transformer/results. You can plot the learning curves using the script nlp-tutorial/text-classification-transformer/plot_results.py. The array _iterations_ in the script must contain the different values for "n_it" used when training.

**ViT Cats and Dogs classification.** Code adapted from this [repository](https://github.com/lucidrains/vit-pytorch). First, you can download the data set [here](https://www.kaggle.com/c/dogs-vs-cats/data), unzip it and save the train and test repositories at sinkformers/vit-pytorch/examples/data. Then you can run 

```bash
cd vit-pytorch
python one_expe.py
cd ..
```


Arguments for one_expe.py can be accessed through 

```bash
cd vit-pytorch
python one_expe.py --help
cd ..
```

Results are saved in the folder vit-pytorch/results. You can plot the learning curves using the script vit-pytorch/plot_results.py. The array _iterations_ in the script must contain the different values for "n_it" used when training.

**ViT MNIST.** The MNIST dataset will be downloaded automatically. 

```bash
cd vit-pytorch
python one_expe_mnist.py
cd ..
```

Arguments for one_expe_mnist.py can be accessed through 

```bash
cd vit-pytorch
python one_expe_mnist.py --help
cd ..
```

Especially, the argument "ps" is the patch size. Results are saved in the folder vit-pytorch/results_mnist. You can plot the learning curves using the script vit-pytorch/plot_results_mnist.py. The array _iterations_ in the script must contain the different values for "n_it" used when training. The array patches_size in the script must contain the different values for "ps" used when training.

Cite
----

If you use this code in your project, please cite::

    Michael E. Sander, Pierre Ablin, Mathieu Blondel, Gabriel Peyré
    Sinkformers: Transformers with Doubly Stochastic Attention
    International Conference on Artificial Intelligence and Statistics (pp. 3515-3530). PMLR.
    https://arxiv.org/abs/2110.11773


