from pathlib import Path
from setuptools import find_packages, setup

setup(
    name='neural-admixture',
    version='1.1.5',
    long_description=(Path(__file__).parent / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    description='Population clustering with autoencoders',
    url='https://github.com/AI-sandbox/neural-admixture',
    author='Albert Dominguez Mantes',
    author_email='adomi@stanford.edu',
    entry_points={
        'console_scripts': ['neural-admixture=entry:main']
    },
    license='CC BY-NC 4.0',
    packages=find_packages('neural_admixture/')+['.'],
    package_dir={"": "neural_admixture"},
    python_requires=">=3.8",
    install_requires=['Cython>=0.29.30',
                      'codetiming==1.3.0',
                      'h5py>=3.1.0',
                      'matplotlib==3.3.4',
                      'numpy>=1.21.0',
                      'pandas==1.2.4',
                      'pandas_plink==2.2.9',
                      'py_pcha==0.1.3',
                      'scikit-allel==1.3.5',
                      'scikit-learn>=0.24.1',
                      'setuptools==50.3.1',
                      'torch>=1.7.1',
                      'wandb>=0.12.17'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.9',
    ],
)
