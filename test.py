# -*- coding: utf-8 -*-
import numpy as np
import requests
import time
from bs4 import BeautifulSoup

def get_number_of_bedrooms(soup):

    soup = str(soup)

    bedroom_index = soup.find('"key">Soverom')
    next_location = "not null"

    while bedroom_index != -1 and next_location != -1:
        bedroom_index += 50
        if soup[bedroom_index-1] != ">":
            print "Invalid common cost: " + soup[bedroom_index:bedroom_index+20]
            next_location = soup[bedroom_index + 1:].find('"key">Soverom')
            bedroom_index += next_location + 1
        else:
            number_of_bedrooms = ""
            char = "not <"
            while char != "<":
                char = soup[bedroom_index:bedroom_index+1]
                if char in ["0","1","2","3","4","5","6","7","8","9"]:
                    number_of_bedrooms += char
                bedroom_index += 1
            try:
                number_of_bedrooms = int(number_of_bedrooms)
                return number_of_bedrooms
            except ValueError:
                print "Could not convert " + number_of_bedrooms + " to int."
    print "Could not find number of bedrooms. Returning 0. Ad code: " + str(ad_code)
    return 0


for ad_code in [91793625,101465867,84178887,97153107,99684638,83062006,105783537,92855596,93119160,100592260,83874654,106382818,106628748,100256538,94908968,107028733,98927968,92645669,86883653,93147689,85478762,108314444,86357385,95402797,95402797,84296898,86609117,105228730,98394733,89211713,103980224,96953273,93113824,81815729,82408152,96242093,88370627,88370627,100973721,102830536,95437312,88994925,105491100,91602905,99810758,81875429,83926699,85942014,105875670,86280182,100578716,103146496,94373135,84895220,102337111,101203019,98123982,105840375,94542353,92959551,95375385,84880367,83639312,85527830,98250719,92651389,102445795,102942261,82723111,92613928,89413084,96319148,96863076,91831760,91413678,108332739,91849974,94467919,96889317,91376237,96241293,96958001,84601289,100987084,92859249,89048083,106109515,96740727,81878094,97685612,98130175,103975090,96336712,104897220,89558273,83850150,84044015,102217774,83437817,107046685,84664546,95983085,98608096,95480026,91587787,83881269,83067770,103951452,95432397,98213160,90898915,93505371,86352503,104478503,99259596,98784329,83867112,85705731,103972817,103364540,89840347]:
    url = "https://www.finn.no/realestate/homes/ad.html?finnkode=" + str(ad_code)

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0'}
    while True:
        try:
            r = requests.get(url, headers=headers)
            break
        except requests.ConnectionError:
            print "Connection denied on URL " + str(url)
            time.sleep(1)

    data = r.text
    soup = BeautifulSoup(data, "lxml")
    print get_number_of_bedrooms(soup)
