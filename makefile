install: all

all:
	echo "installing..."
	python3 setuptools_script.py bdist_wheel
	python3 -m pip install dist/*.whl
	pip3 install GuidedFilters

clean:
	echo "cleaning..."
	pip3 uninstall -y GuidedFilters
	rm -rf GuidedFilters.egg-info/
	rm -rf build/
	rm -rf dist/

	
