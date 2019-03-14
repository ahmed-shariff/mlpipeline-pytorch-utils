# Simple helper script to reinstall and test the package
rm -rf dist
rm -rf build
python setup.py sdist bdist_wheel
pip uninstall mlpipeline_torch_utils -y
pip install dist/*
