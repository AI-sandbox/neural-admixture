from setuptools import find_packages, setup


setup(
    name='neural-admixture',
    version='0.5.1',
    description='Population clustering with autoencoders',
    url='https://github.com/AI-sandbox/neural-admixture',
    author='Albert Dominguez Mantes',
    author_email='adomi@stanford.edu',
    scripts=['./neural-admixture/neural-admixture'],
    license='CC BY-NC 4.0',
    packages=find_packages('neural-admixture/')+['.'],
    package_dir={"": "neural-admixture"},
    include_package_data=True,
    install_requires=['codetiming==1.3.0',
                      'h5py==3.1.0',
                      'matplotlib==3.3.4',
                      'pandas==1.2.4',
                      'pandas_plink==2.2.9',
                      'scikit-allel==1.3.5',
                      'scikit-learn==0.24.1',
                      'setuptools==50.3.1',
                      'torch==1.7.1',
                      'wandb==0.10.21'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
