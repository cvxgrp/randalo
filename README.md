# ALO Library

## Installation

In a folder run the following:
```
git clone git@github.com:cvxgrp/alo.git
cd alo
python -m venv venv
. venv/bin/activate

pip install wheel
pip install torch numpy scipy matplotlib

cd ..

git clone git@github.com:cvxgrp/SURE-CR.git
cd SURE-CR
git checkout xtrace
pip install -e .

cd ..

git clone git@github.com:cvxgrp/torch_linops.git
cd torch_linops
pip install -e .

cd ../alo
pip install -e .
```
