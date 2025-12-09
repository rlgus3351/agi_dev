from meqk import debug_meqk

meqk_data = {
    "1":  "11",
    "2":  "22",
    "3":  "1",
    "4":  "4",
    "5":  "4",
    "6":  "3",
    "7":  "3",
    "8":  "1",
    "9":  "2",
    "10": "20.5",
    "11": "2",
    "12": "3",
    "13": "1",
    "14": "1",
    "15": "2",
    "16": "2",
    "17": "23",
    "18": "11",
    "19": "3",
}

result = debug_meqk(meqk_data)

from pprint import pprint
pprint(result)
