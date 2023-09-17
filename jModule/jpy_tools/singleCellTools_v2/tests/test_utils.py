import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as ss
from ...otherTools import F
from .. import utils

def test_ad2df():
    ar = np.random.randint(0, 100, size=(100, 4))
    df = pd.DataFrame(ar, columns=list('ABCD'), index=[str(x) for x in range(100)])
    ad = sc.AnnData(df)
    df2 = utils.ad2df(ad)
    assert (df2 == df).all().all()

def test_ad2df_sparse():
    ar = np.random.randint(0, 100, size=(100, 4))
    df = pd.DataFrame(ar, columns=list('ABCD'), index=[str(x) for x in range(100)])
    ad = sc.AnnData(df)
    ad.X = ss.csc_matrix(ad.X)
    df2 = utils.ad2df(ad)
    assert (df2 == df).all().all()
    assert pd.api.types.is_sparse(df2.iloc[0])

def test_initLayer():
    ar = np.random.randint(0, 100, size=(100, 4))
    df = pd.DataFrame(ar, columns=list('ABCD'), index=[str(x) for x in range(100)])
    ad = sc.AnnData(df)
    ad.X = ss.csc_matrix(ad.X)
    utils.initLayer(ad)
    assert 'raw' in ad.layers
    assert 'normalize_log' in ad.layers
    assert (np.round((np.exp(ad.X.A) - 1).sum(1)) == 1e4).all()

    ad.X = ad.layers['raw'].copy()
    utils.initLayer(ad, logbase=2)
    assert (np.round((np.exp2(ad.X.A) - 1).sum(1)) == 1e4).all()

def test_getOverlap():
    ad1 = sc.AnnData(pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'), index=[str(x) for x in range(100)]))
    ad2 = sc.AnnData(pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABFG'), index=[str(x) for x in range(100)]))
    ad1, ad2 = utils.getOverlap(ad1, ad2)
    assert ad1.var_names.tolist() == ['A', 'B']
    assert ad2.var_names.tolist() == ['A', 'B']

def test_splitAdata():
    ad = sc.AnnData(pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'), index=[str(x) for x in range(100)]))
    ad.obs['batch'] = np.random.choice(['a', 'b'], size=100)
    i = 0
    for _ad in utils.splitAdata(ad, 'batch', axis=0):
        i += _ad.shape[0]
    assert i == 100

def test_testAllCountIsInt():
    ad = sc.AnnData(pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'), index=[str(x) for x in range(100)]))
    utils.testAllCountIsInt(ad)
    utils.initLayer(ad)
    try:
        utils.testAllCountIsInt(ad)
    except AssertionError:
        pass
    else:
        assert False

def test_getPartialLayersAdata():
    ad = sc.AnnData(pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'), index=[str(x) for x in range(100)]))
    utils.initLayer(ad)
    ad.obs['aa'] = 1
    ad.obs['bb'] = 2
    ad_1 = utils.getPartialLayersAdata(ad, None, ['aa'])
    assert len(ad_1.layers.keys())==0
    assert 'aa' in ad_1.obs.columns
    assert not ('bb' in ad_1.obs.columns)

def test_mergeData():
    ad = sc.AnnData(pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'), index=[str(x) for x in range(100)]))
    utils.initLayer(ad)
    ad.obs['batch'] = np.random.choice(['a', 'b'], size=100)
    ad_merge = utils.mergeData(ad, 'batch')
    assert ad[ad.obs.eval("batch == 'a'")].layers['raw'].sum() == ad_merge.X[0, :].sum()
    assert ad[ad.obs.eval("batch == 'b'")].layers['raw'].sum() == ad_merge.X[1, :].sum()
