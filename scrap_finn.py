# -*- coding: utf-8 -*-

import numpy as np
import time
import random
import alva_io
import pandas as pd
from multiprocessing import Pool
from bs4 import BeautifulSoup, UnicodeDammit
import requests
#import warnings
#warnings.filterwarnings("ignore")

REFURBISHMENT_WORDS = ["oppussingsbehov", "oppussingsobjekt", "oppgraderingsbehov", "oppussing må påregnes"]
POOL_SIZE = 10


def get_soup(ad_code):

    if random.random() > 0.99:
        print "100. Ad code: " + str(ad_code)

    url = "https://www.finn.no/" + str(ad_code)

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0'}
    while True:
        try:
            r = requests.get(url, headers = headers)
            break
        except requests.ConnectionError:
            print "Connection denied on URL " + str(url)
            time.sleep(1)
    """
    unicode_str = r.text.encode("utf-8")
    encoded_str = unicode_str.decode("ascii", 'ignore')
    soup = BeautifulSoup(r.text,"html.parser")
    """
    data = r.text
    soup = BeautifulSoup(data,"lxml")
    title = get_title(soup)

    title = title.replace(";",".")

    build_year = get_build_year(soup)

    return (title, build_year)

def get_title(soup):

    text = soup.find("title")

    text = str(text)

    text = text.strip("<title>")
    text = text.strip("</title>")

    return text

def get_build_year(soup):

    soup = str(soup)

    build_year_index = soup.find("Byggeår")
    next_location = "not null"

    while build_year_index != -1 and next_location != -1:
        try:
            build_year_index += 45
            return int(soup[build_year_index:build_year_index+4])
        except ValueError:
            next_location = soup[build_year_index + 1:].find("Byggeår")
            build_year_index += next_location + 1

    return np.nan

def needs_refurbishment(ad_code):

    string = get_title(get_soup(ad_code))
    for word in REFURBISHMENT_WORDS:
        if word in string.lower():
            return True
    return False

def get_minutes_and_seconds(time):

    seconds = int(round(time))
    minutes = seconds // 60
    remaining_seconds = seconds - minutes * 60

    return str(minutes) + " minutes and " + str(remaining_seconds) + " seconds."

soups_created = False

if __name__ == "__main__":

    start = time.time()

    virdi_augmented = pd.read_csv("C:/Users/tobiasrp/data/virdi_augmented.csv", index_col=0)

    #virdi_augmented = virdi_augmented.sample(frac = 1)
    #virdi_augmented = virdi_augmented[:int(0.01*len(virdi_augmented))]

    ad_codes = [int(i) for i in virdi_augmented.ad_code.tolist()]

    print "--------------------------------------------------"
    print get_soup(ad_codes[0])
    print "--------------------------------------------------"

    print "MAIN OK"
    p = Pool(POOL_SIZE)
    soups = p.map(get_soup, ad_codes)
    soups_created = True

if soups_created:
    build_years = []
    titles = []
    for s in soups:
        titles.append(s[0])
        build_years.append(s[1])
    """
    for s in soups:
        titles.append(get_title(s))
        build_years.append(get_build_year(s))
    """
    title_and_ad_codes = pd.DataFrame({'ad_code':ad_codes, 'title': titles, 'build_year': build_years})
    virdi_augmented = pd.merge(virdi_augmented, title_and_ad_codes, how="left", on="ad_code")
    alva_io.write_to_csv(virdi_augmented, "C:/Users/tobiasrp/data/virdi_augmented_with_title.csv")

    print "Terminated in " + get_minutes_and_seconds(time.time() - start)