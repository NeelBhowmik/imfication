# cnn-material-classifiation

Tested using Python >= 3.8.x, [PyTorch >= 1.7](https://pytorch.org/).

## Architectures:
![Architectures](images/arch.png)
---
## Installation
The code is tested on Ubuntu 20.04.  
### Requirements for Deskop/Laptop
1. Linux (Ubuntu >= 20.04 distribution)
2. CUDA >= 11.0, cuDNN >= 8.1.0
3. Python ≥ 3.8

### Steps
0. [Optional] create a new virtual environment.

    ~~~
    sudo apt update
    sudo apt install python3-dev python3-pip
    ~~~
    And activate the environment.

    ~~~
    source ./venv/bin/activate # sh, bash, ksh, or zsh
    ~~~
1. First clone the repository:
    ~~~
    git clone https://github.com/NeelBhowmik/cnn-material-classifiation.git
    ~~~

2. Install **pytorch >= 1.7.0** with torchvision (that matches the PyTorch installation - [link](https://pytorch.org/)).

3. Install the requirements

    ~~~
    pip3 install -r requirements.txt
    ~~~
---
## Usage:

1. For traning:
    ~~~
    train.py [-h] [--db DB] [--dbpath DBPATH] [--dbsplit DBSPLIT]        
                  [--datatype DATATYPE] [--net NET] 
                  [--optim OPTIM] [--ft] [--pretrained] [--lr LR] 
                  [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] 
                  [--batch BATCH] [--ichannel ICHANNEL] 
                  [--isize ISIZE] [--epoch EPOCH] 
                  [--save_freq SAVE_FREQ] [--cpu] 
                  [--workers WORKERS] [--outf OUTF]

    optional arguments:
        -h, --help            show this help message and exit
        --db DB               dataset name
        --dbpath DBPATH       Specify the dataset directory path
        --dbsplit DBSPLIT     Specify the dataset dataset split
        --datatype DATATYPE   Specify the datatype {hlz, xglcm}
        --net NET             Select the network {simnet, resnet50,...}
        --optim OPTIM         Select optimizer {SGD, Adam}
        --ft                  If true - only update the reshaped layer params. If flase - traning from scratch
        --pretrained          Use pretrained network.
        --lr LR               initial learning rate for opimisation
        --momentum MOMENTUM   momentum term of optimisation
        --weight_decay WEIGHT_DECAY
                                weight decay term of optimisation
        --batch BATCH         input training batch size
        --ichannel ICHANNEL   input data channel no
        --isize ISIZE         input data size
        --epoch EPOCH         number of traning epoch
        --save_freq SAVE_FREQ
                                save model weight interval
        --cpu                 If selected will run on CPU
        --workers WORKERS     number of data loading workers
        --outf OUTF           A directory path to save model output
    ~~~

2. For testing:
    ~~~
    test.py [-h] [--db DB] [--dbpath DBPATH] [--dbsplit DBSPLIT] 
                 [--datatype DATATYPE] [--net NET] [--ft]
                 [--pretrained] [--weight WEIGHT] 
                 [--batch BATCH] [--ichannel ICHANNEL] 
                 [--isize ISIZE] [--cpu]
                 [--workers WORKERS] [--statf STATF]

    optional arguments:
        -h, --help           show this help message and exit
        --db DB              dataset name
        --dbpath DBPATH      Specify the dataset directory path
        --dbsplit DBSPLIT    Specify the dataset dataset split
        --datatype DATATYPE  Specify the datatype {hlz, xglcm}
        --net NET            Select the network {simnet, resnet50,...}
        --ft                 If true - only update the reshaped layer params. If flase - traning from scratch
        --pretrained         Use pretrained network.
        --weight WEIGHT      path to model weight file
        --batch BATCH        input testing batch size
        --ichannel ICHANNEL  input data channel no
        --isize ISIZE        input data size
        --cpu                If selected will run on CPU
        --workers WORKERS    number of data loading workers
        --statf STATF        A directory path to save statistics
    ~~~