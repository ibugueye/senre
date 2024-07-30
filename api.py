from flask import Flask, jsonify
import socket

import pandas as pd

app = Flask(__name__)

# Charger les données
df = pd.read_csv('senre_data.csv',on_bad_lines='skip')


@app.route('/api/chiffre_affaires')
def get_chiffre_affaires():
    ca_data = df[df['Section'] == 'chiffre_affaires_zones'][['Zones', '2018']].to_dict('records')
    return jsonify(ca_data)

@app.route('/api/repartition_geographique')
def get_repartition_geographique():
    repartition = df[df['Section'] == 'chiffre_affaires_zones'][['Zones', '2018']].to_dict('records')
    return jsonify(repartition)

@app.route('/api/structure_portefeuille')
def get_structure_portefeuille():
    structure = df[df['Section'] == 'chiffre_affaires_branches'][['Branches', '2018']].to_dict('records')
    return jsonify(structure)

if __name__ == '__main__':
    port = 5001
    while True:
        try:
            app.run(debug=True, port=port)
            break
        except socket.error as e:
            if e.errno == 98:  # Address already in use
                port += 1
            else:
                raise