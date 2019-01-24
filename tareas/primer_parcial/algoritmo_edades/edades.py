print('                          leeme                           ')
print('------------------------------------------------------------')
print('|                    Descubro tu edad                      |')
print('|               Introduce lo que se te pida                |')
print('| Cuando ponga Y/N pon una Y si estas de acuerdo y N si no |')
print('|      Todos los parametros son con el sistema metrico     |')
print('|        En la estatura se utilizara el formato 1.xx       |')
print('------------------------------------------------------------')
puntuacion = 0
nombre = input('Introduce tu nombre: ')
sexo = input('Introduce tu sexo de nacimiento M/H:')
if sexo == 'h':
    estatura = 1.60
else:
    estatura = 1.50
estatura_usuario = input('Introduce tu estatura: ')
if float(estatura_usuario) > estatura:
     puntuacion += 1
print('Cual es tu grado maximo de estudios')
print('Kinder                            1')
print('Primaria                          2')
print('Secundaria                        3')
print('Preparatoria                      4')
print('Universidad                       5')
print('Mas                               6')
Escuela = input('Selecciona una opcion: ')
puntuacion += int(Escuela)
tu_edad = {
    (puntuacion < 4) : '0  a 16',
    (puntuacion < 6) : '16 a 20'
}
print('Tu edad esta entre {}'.format(tu_edad))
