from setuptools import setup, find_packages

# Read the contents of your README file with UTF-8 encoding
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='DepoisonAI',
    version='1.0.0',
    author='Yash Hambarde',
    description='An AI system to automatically detect and reverse data poisoning attacks.',
    long_description=long_description, # Use the variable here
    long_description_content_type='text/markdown',
    url='https://github.com/theyashhambarde/DepoisonAI',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security',
    ],
    python_requires='>=3.9',
    install_requires=open('requirements.txt').read().splitlines(),
    include_package_data=True,
)
