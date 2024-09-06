# ALO Library

## Installation

In a folder run the following:

```
git clone git@github.com:cvxgrp/randalo.git
cd randalo

# create a new environment with torch & friends (could also use conda or similar)
python -m venv venv
. venv/bin/activate

pip install wheel
pip install torch numpy scipy matplotlib

pip install git+ssh://git@github.com/cvxgrp/SURE-CR.git@xtrace
pip install git+ssh://git@github.com/cvxgrp/torch_linops.git
pip install -e .
```
