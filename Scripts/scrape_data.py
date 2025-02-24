import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_airline_codes"

tables = pd.read_html(url)

airline_codes_table = tables[0]

print(airline_codes_table.head())

airline_codes_table.to_csv("../Data/airline_codes.csv", index=False)

# airport codes:

letter_ascii = 65
base_url = "https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_"
for i in range(26):
    url = base_url + chr(letter_ascii)
    tables = pd.read_html(url)
    airport_codes_table = tables[0]
    airport_codes_table.to_csv("../Data/airport_codes.csv", index=False, mode='a')
    letter_ascii += 1
