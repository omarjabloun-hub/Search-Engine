
## Getting started

### Setting up environment
Create virtual Environment & Activate it

``` bash
python -m venv venv
 
```
In the python environment run:

``` bash
venv\Scripts\activate  
pip install -r .\requirements.txt
python script.py
```
This will activate the environment and install dependencies

### Run the FastAPI server

``` bash
uvicorn app:app --reload
```


