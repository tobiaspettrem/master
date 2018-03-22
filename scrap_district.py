
from bs4 import BeautifulSoup, UnicodeDammit
import requests
import warnings
warnings.filterwarnings("ignore")

def get_raw_text(address):
    url = "https://www.oslo.kommune.no/xmlhttprequest.php?service=district.getDistricts&term=" + address

    r = requests.get(url)
    data = r.text

    soup = BeautifulSoup(data,"lxml")
    text = soup.find("p")

    return text

def get_bydel(raw_text):
    string = str(raw_text)
    hit = string.split(",")[0]
    right_side = hit.split("(")[1]
    left_side = right_side.split(")")[0]
    bydel = left_side.replace("Bydel ","")
    return bydel

def get_bydel_kode(raw_text):
    string = str(raw_text)
    try:
        hit = string.split(",")[1]
    except IndexError:
        return False
    right_side = hit.split(":")[1]
    bydel_kode = right_side[1:4]
    return bydel_kode

