from __future__ import print_function
from alphatools.ics.ics_scheme import make_sector_classifier

import click
from os import path

@click.group()
def main():
    pass

@main.command()
def ingest():
    print('mapping sectors and industries...')
    make_sector_classifier()
    
if __name__ == '__main__':
    main()
