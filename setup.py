import setuptools

__version__='0.1.a.1'

# to build and install:
# python setup.py
# pip install dist/mlpipeline-*-py3-none-any.whl
#
# or
#
# pip install -e . --upgrade

setuptools.setup(
    name="mlpipeline_torch_utils",
    version=__version__,
    author='Ahmed Shariff',
    author_email='shariff.mfa@outlook.com',
    packages=setuptools.find_packages(),
    description='Utils for pytorch to be used with https://github.com/ahmed-shariff/ml-pipeline',
    long_description=open('README.md').read(),
    url='',
    install_requires=['easydict>=1.8',
                      'mlpipeline>=1.1a3.post8'],
    # entry_points={
    #     'console_scripts':[
    #         'mlpipeline=mlpipeline.pipeline:main',
    #         '_mlpipeline_subprocess=mlpipeline._pipeline_subprocess:main'
    #         ]
    #     },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        
    ]
)
