import numpy as np
import pandas as pd
from numpy import where, zeros

from zipline.data import bundles
from zipline.pipeline.classifiers import Classifier
from zipline.utils.numpy_utils import int64_dtype, object_dtype
    
from os import path


class SICMajorIndustry(Classifier):

    dtype = int64_dtype
    window_length = 0
    inputs = ()
    missing_value = -1
    
    def __init__(self):
        self.data = np.load('sic_major.npy')
        self.names = None
    def _compute(self, arrays, dates, assets, mask):
        return where(
            mask,
            self.data[assets],
            self.missing_value,
        )
    
class Sector(Classifier):

    dtype = int64_dtype
    window_length = 0
    inputs = ()
    missing_value = -1
    
    def __init__(self):
        self.data = np.load(
            path.join(path.dirname(__file__), 'sectors.npy')
        )
        self.names = pd.read_csv(
            path.join(path.dirname(__file__), 'sector_names.csv'),
            header=None,
            index_col=0,
            names=['Sector_Name']
        )

    def _compute(self, arrays, dates, assets, mask):
        return where(
            mask,
            self.data[assets],
            self.missing_value,
        )


class SubIndustry(Classifier):

    dtype = int64_dtype
    window_length = 0
    inputs = ()
    missing_value = -1
    
    def __init__(self):
        self.data = np.load('industries.npy')
        self.names = pd.read_csv(
            'industry_names.csv',
            header=None,
            index_col=0,
            names=['Industry_Name']
        )

    def _compute(self, arrays, dates, assets, mask):
        return where(
            mask,
            self.data[assets],
            self.missing_value,
        )


def make_sector_classifier(
        bundle='quandl',
        infile='../data/profiles_20170918.csv'):
    """
    For a given bundle, create the .npy Sector and Industry classifier
    files.
    """
    bundle_data = bundles.load(bundle)

    df_p = pd.read_csv(infile)

    labels_sector, uniques_sector = pd.factorize(df_p['sector'])
    labels_industry, uniques_industry = pd.factorize(df_p['industry'])
    
    tickers = bundle_data.asset_finder.lookup_symbols(
        df_p['quandl_sym'], as_of_date=None)
    
    sids = [asset.sid for asset in tickers]
    max_sid = np.max(bundle_data.asset_finder.sids)

    sectors = np.full(np.max(max_sid)+1, -1, np.dtype('int64'))
    industries = np.full(np.max(max_sid)+1, -1, np.dtype('int64'))

    sectors[sids] = labels_sector
    industries[sids] = labels_industry

    np.save('sectors', sectors)
    np.save('industries', industries)

    pd.DataFrame(data=uniques_sector.tolist()).to_csv(
        'sector_names.csv', header=False)
    pd.DataFrame(data=uniques_industry.tolist()).to_csv(
        'industry_names.csv', header=False)

    return True

def make_SIC_classifier(
        bundle='quandl',
        infile='../data/profiles_20170918.csv'):

    bundle_data = bundles.load(bundle)

    df_p = pd.read_csv(infile)
    df_cik = pd.read_csv('../data/cik_ticker_09152017.csv', sep='|')
    df_cik['SIC'] = df_cik['SIC'].fillna(-1).astype(np.int64).astype(str)
    df_cik['SIC_MajorIndustry'] = df_cik['SIC'].str[:2]
    df_cik['SIC_SubClassification'] = df_cik['SIC'].str[:3]
    df_cik['SIC_Specialization'] = df_cik['SIC'].str[:4]

    df_cik_select = df_cik.loc[df_cik.Ticker.isin(df_p['quandl_sym'])]
    tickers = bundle_data.asset_finder.lookup_symbols(df_cik_select['Ticker'], as_of_date=None)
    sids = [asset.sid for asset in tickers]
    max_sid = np.max(bundle_data.asset_finder.sids)

    major = np.full(max_sid+1, -1, np.dtype('int64'))
    subclass = np.full(max_sid+1, -1, np.dtype('int64'))
    specialize = np.full(max_sid+1, -1, np.dtype('int64'))
    
    major[sids] = df_cik_select['SIC_MajorIndustry'].astype(np.int64)
    subclass[sids] = df_cik_select['SIC_SubClassification'].astype(np.int64)
    specialize[sids] = df_cik_select['SIC_Specialization'].astype(np.int64)

    np.save('sic_major', major)
    np.save('sic_subclass', subclass)
    np.save('sic_specialize', specialize)


    
if __name__ == '__main__':
    make_sector_classifier()
    make_SIC_classifier()
