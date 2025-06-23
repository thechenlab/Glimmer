from setuptools import setup, find_packages

setup(
    name='glimmer-st',
    version='0.1.0',
    author='Qiyu Gong',
    author_email='gongqiyu@broadinstitute.org',
    description='A Unified Framework for Graph-Based Representation of Spatial Structures Across Modalities and Scales',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thechenlab/Glimmer',
    packages=find_packages(),
    install_requires=[
        'numpy==2.2.6',
        'pandas>=1.5.0',
        'scipy==1.15.3',
        'torch==2.7.0',
        'tqdm==4.67.1',
        'scikit-learn==1.5.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
