# Simple helper script to reinstall and test the package
python setup.py sdist bdist_wheel
pip uninstall mlpipeline_torch_utils -y
rm -rf dist
rm -rf build
pip install dist/*
