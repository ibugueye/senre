import csv

data = {
    "chiffre_affaires_zones": [
        ["Section", "Zones", "2018", "Pourcentage_2018", "2017", "Pourcentage_2017", "Variation"],
        ["chiffre_affaires_zones", "AFRIQUE AUSTRALE", 42275677, 0.26, 72825865, 0.52, -41.95],
        ["chiffre_affaires_zones", "AFRIQUE CENTRALE", 423419940, 2.58, 171800220, 1.23, 146.46],
        ["chiffre_affaires_zones", "AFRIQUE L'EST O. INDIEN", 337752335, 2.06, 1088674469, 7.78, -68.98],
        ["chiffre_affaires_zones", "AFRIQUE DE L'OUEST", 13341334820, 81.21, 10807307785, 77.19, 23.45],
        ["chiffre_affaires_zones", "AFRIQUE DU NORD", 1299947817, 7.91, 1100103737, 7.86, 18.17],
        ["chiffre_affaires_zones", "ASIE", 781278249, 4.76, 453949273, 3.24, 72.11],
        ["chiffre_affaires_zones", "MOYEN ORIENT", 201115398, 1.22, 306565756, 2.19, -34.40],
        ["chiffre_affaires_zones", "EUROPE", 988403, 0.01, -94007, 0.00, -1151.41],
        ["chiffre_affaires_zones", "TOTAL", 16428112641, 100, 14001133098, 100, 17.33]
    ],
    "chiffre_affaires_branches": [
        ["Section", "Branches", "2018", "Pourcentage_2018", "2017", "Pourcentage_2017", "Variation"],
        ["chiffre_affaires_branches", "VIE", 1033416886, 6.29, 2024559807, 14.46, -48.96],
        ["chiffre_affaires_branches", "INCENDIE", 5640869619, 34.34, 4511719390, 32.22, 25.03],
        ["chiffre_affaires_branches", "TRANSPORTS", 1353554278, 8.24, 1178411625, 8.42, 14.86],
        ["chiffre_affaires_branches", "AUTOMOBILE", 3023508564, 18.40, 2019153364, 14.42, 49.74],
        ["chiffre_affaires_branches", "RISQUES TECHNIQUES", 1200898531, 7.31, 1002120006, 7.16, 19.84],
        ["chiffre_affaires_branches", "RISQUES DIVERS", 4081868025, 24.85, 3215808338, 22.97, 26.93],
        ["chiffre_affaires_branches", "AVIATION", 93996737, 0.57, 49360569, 0.35, 90.43],
        ["chiffre_affaires_branches", "TOTAL", 16428112641, 100, 14001133098, 100, 17.33]
    ],
    "indicateurs_financiers": [
        ["Section", "Annee", "Capital_Social", "Capitaux_Propres", "Resultat_Net", "ROE", "Chiffre_Affaires", "Retrocession", "Taux_Frais_Generaux", "Ratio_Combine_Net", "Placements", "Provisions_Techniques", "Taux_Couverture", "Effectif", "Taux_Encadrement", "Chiffre_Affaires_Effectif"],
        ["indicateurs_financiers", 2014, 6585020, 7646768, 752610, 9.84, 17029921, 1200489, 5.62, 97, 38275974, 22436693, 171, 17, 47, 1001760],
        ["indicateurs_financiers", 2015, 6585020, 8007888, 761121, 9.50, 16398887, 1521365, 6.57, 98, 39783670, 21583960, 184, 21, 38, 780899],
        ["indicateurs_financiers", 2016, 6585020, 8547092, 905039, 10.59, 14544231, 1136267, 8.02, 91, 42632720, 21767230, 196, 18, 44, 808013],
        ["indicateurs_financiers", 2017, 6585020, 9513876, 1394810, 14.66, 14001133, 1612309, 8.37, 88, 26760245, 20723063, 129, 18, 39, 777841],
        ["indicateurs_financiers", 2018, 6585020, 10898977, 1385101, 12.71, 16428113, 1493160, 7.66, 98, 28611734, 24906680, 115, 19, 47, 864638]
    ],
    "ratios_prudentiels": [
        ["Section", "Ratio", "Valeur"],
        ["ratios_prudentiels", "Primes Nettes / Fonds Propres", "137.31%"],
        ["ratios_prudentiels", "Provisions Techniques / Fonds Propres", "228.52%"],
        ["ratios_prudentiels", "Provisions Techniques+Fonds Propres / Primes Nettes", "239.25%"],
        ["ratios_prudentiels", "RATIOS COMBINES", "98.21%"]
    ]
}

# Écrire les données dans un fichier CSV
with open('senre_data.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for section, rows in data.items():
        writer.writerows(rows)

