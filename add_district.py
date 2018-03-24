
import pandas as pd, numpy as np, alva_io, ast
from scrap_district import get_bydel_kode, get_raw_text
import time

district_id_to_code = {1:"bgo",2:"bga",3:"bsa",4:"bsh",5:"bfr",6:"bun",7:"bva",8:"bna",
9:"bbj",10:"bgr",11:"bsr",12:"bal",13:"bos",14:"bns",15:"bsn",16:"SEN",17:"MAR"}

def add_bydel(df):

    print "Adding districts."
    districts_list = []
    try:
        street_dict_file = open("street_district_mapping.txt","a+")
        street_to_district = initiate_street_dict(street_dict_file)
    except IOError:
        street_to_district = dict()
    invalid_streets = []
    start = time.time()

    size = float(len(df))
    old_progress, new_progress = 0,0

    for i in range(len(df)):
        row = df.iloc[i]
        district_number = row.bydelsnr
        if np.isnan(district_number):
            street = str(row["eiendomadressegatenavn"])
            if street in street_to_district:
                bydel = street_to_district[street]
            else:
                bydel_kode = get_bydel_kode(get_raw_text(street))
                if bydel_kode:
                    bydel = pd.Series(bydel_kode)
                    street_to_district[street] = bydel
                    street_dict_file.write(street + ":" + bydel_kode + "\n")
                else:
                    invalid_streets.append(street)
                    bydel = pd.Series("NA")
        else:
            bydel = pd.Series(district_id_to_code[district_number])

        districts_list.append(bydel)
        new_progress = round(i / size,2)
        if old_progress != new_progress:
            if (int(100*new_progress)) % 10 == 0:
                print str(int(100*new_progress)) + "%",
            else:
                print "|",
        old_progress = new_progress

    street_dict_file.close()

    districts = pd.Series(b[0] for b in districts_list)

    print("Finished adding districts")
    print("Time elapsed: " + str(time.time() - start))
    for s in set(invalid_streets):
        print s

    df = df.reset_index(drop=True) # drop=True removes old index instead of inserting it as a column
    df = df.assign(bydel_code = districts)

    return df

def initiate_street_dict(file):
    d = dict()
    for line in file.readlines():
        split_line = line.strip().split(":")
        key = split_line[0]
        value = split_line[1]
        d[key] = pd.Series(value)
    return d