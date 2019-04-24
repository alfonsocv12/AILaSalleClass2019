import json, string

array_json = []
counter = list(range(2))
count = 1
for count in counter:
    vector = {}
    vector['id'] = count
    vector['mensaje'] = raw_input('introduce el mensaje: ')
    vector['positivo_negativo'] = input('introduce si es positivo_negativo')
    array_json.append(vector)
    count += 1

with open('amlo.txt','w') as amlo:
    json.dump(array_json, amlo)
