from setuptools import setup, find_packages

setup(
    name='alphatools',
    version='0.11',
    description='Quant finance resarch tools',
    author='Jonathan Larkin',
    author_email='jonathan.r.larkin@gmail.com',
    url = "https://github.com/marketneutral/alphatools",
    download_url = "https://github.com/marketneutral/alphatools/archive/0.11.tar.gz",
    packages=find_packages(),
    python_requires='>2.7, <3.0',
    install_requires=[
        'zipline<=1.3',
        'alphalens',
        'ipykernel'
    ],
    entry_points={
        'console_scripts': [
            'alphatools = alphatools.__main__:main',
        ]
    }
)
