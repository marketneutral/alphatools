from __future__ import print_function
from alphatools.ics.ics_scheme import make_sector_classifier

import click
import subprocess
import sys

from os import path
import zipline

this_path = path.dirname(__file__)

@click.group()
def main():
    pass

@main.command()
def get_blaze():
    req = path.join(this_path, 'requirements_blaze.txt')
    print(req)
    subprocess.call([sys.executable, "-m", "pip", "install", "-r" + req])

# Example
if __name__ == '__main__':
    install('argh')

@main.command()
def ingest():
    print('mapping sectors and industries...')
    make_sector_classifier()
    
if __name__ == '__main__':
    main()
