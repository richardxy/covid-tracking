install:
	pip install -r requirements.txt

test:
	#python -m pytest test_gcli.py
	python -m pytest -vv test/test_atop.py
	#python -m pytest --nbval notebook.ipynb

lint:
	pylint --disable=R,C atop.py

all: 
	install lint test
