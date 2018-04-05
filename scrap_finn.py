# -*- coding: utf-8 -*-

import random
import alva_io
import pandas as pd
from multiprocessing import Pool
from bs4 import BeautifulSoup, UnicodeDammit
import requests
import warnings
warnings.filterwarnings("ignore")

REFURBISHMENT_WORDS = ["oppussingsbehov", "oppussingsobjekt", "oppgraderingsbehov", "oppussing må påregnes"]


def get_title(ad_code):

    if random.random() > 0.99:
        print "100. Ad code: " + str(ad_code)

    url = "https://www.finn.no/" + str(ad_code)

    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0' }
    r = requests.get(url, headers = headers)

    data = r.text

    soup = BeautifulSoup(data,"lxml")

    text = str(soup.find("title"))
    text = text.strip("<title>")
    text = text.strip("</title>")

    return text

def needs_refurbishment(ad_code):

    string = get_title(ad_code)
    for word in REFURBISHMENT_WORDS:
        if word in string.lower():
            return True
    return False

def get_title_series(ad_code_list):
    title_list = []
    for ad_code in ad_code_list:
        title_list.append(get_title(ad_code))

    title_series = pd.Series(title_list)
    return title_series

virdi_augmented = pd.read_csv("C:/Users/Tobias/data/virdi_augmented.csv",index_col=0)

AD_CODE_SIZE = 50
POOL_SIZE = 10

ad_codes = [int(i) for i in virdi_augmented.ad_code.tolist()]
# ad_codes = ad_codes[:AD_CODE_SIZE]
titles_created = False


if __name__ == "__main__":
    print "MAIN OK"
    p = Pool(POOL_SIZE)
    titles = p.map(get_title, ad_codes)
    titles_created = True

if titles_created:
    title_and_ad_codes = pd.DataFrame({'ad_code':ad_codes, 'title': titles})
    virdi_augmented = pd.merge(virdi_augmented, title_and_ad_codes, how="left", on="ad_code")
    alva_io.write_to_csv(virdi_augmented, "C:/Users/Tobias/data/virdi_augmented_with_title.csv")