

from bs4 import BeautifulSoup, UnicodeDammit
import requests
import warnings
warnings.filterwarnings("ignore")

REFURBISHMENT_WORDS = ["oppussingsbehov", "oppussingsobjekt", "oppgraderingsbehov", "oppussing maa paaregnes"]

def get_raw_text(ad_code):
    url = "https://www.finn.no/" + str(ad_code)

    r = requests.get(url)
    data = r.text

    soup = BeautifulSoup(data,"lxml")

    text = soup.find("title")

    print text

    return text


def needs_refurbishment(raw_text):
    string = str(raw_text)

    hit = "agOIMGORISNoAIGNoING"

    print hit.lower()

    hit = str(string.split(">")[2])

    print hit.lower()

    for word in REFURBISHMENT_WORDS:
        if word in hit:
            return True
    return False


print (needs_refurbishment(get_raw_text(96724954)))
print (needs_refurbishment(get_raw_text(116354088)))
