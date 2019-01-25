'''
Imports
'''
import sys

'''
Empieza el programa
'''
print('                          leeme                           ')
print('------------------------------------------------------------')
print('|                    Descubro tu edad                      |')
print('|               Introduce lo que se te pida                |')
print('| Cuando ponga S/N pon una Y si estas de acuerdo y N si no |')
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
print('Cual es tu estado civil')
print('Casado/Soltero/viudo')
estado_civil = input('C/S/V:')
if puntuacion < 5:
    '''
    Posiblemente adulto mayor
    '''
    nietos = input('Tiene nietos S/N: ')

    if nietos == 'S':
        '''
        Su edad minima es de 30 años
        '''
        if estado_civil == 'V':
            '''
            Muy probable que sea adulto mayor
            '''
            print('Por sus datos calculo que su edad es 60 o Mas ')
            sys.exit(0)
        elif estado_civil == 'C':
            '''
            Muy probable que sea adulto mayor
            '''
            print('Por sus datos calculo que su edad es 60 o Mas ')
            sys.exit(0)
    else:
        '''
        Probablemente un niño
        '''
        casa = input('Vives en la casa de tus padres Y/N: ')
        if casa == 'Y':
            if puntuacion < 2:
                print('Por los datos que me diste calculo que eres un niño')
                sys.exit(0)
            else:
                '''
                Posiblemente un adolecente:
                '''
                continuas_estudiando = input('Sigues estudiando S/N: ')
                if continuas_estudiando == 'S':
                    '''
                    Muy Posiblemente un adolecente
                    '''
                    print('por la informacion que me diste creo que tu edad esta entre 10 y 15')
                    sys.exit(0)
elif Escuela == 5:
    '''
    Probablemente adulto
    '''
    hijos = input('Tienes hijos S/N: ').upper()
    if hijos == 'S':
        '''
        Probablemente mayor a 22
        '''
        nietos = input('Tienes nietos S/N: ').upper()
        if nietos == 'S':
            print('Probablemente tienes de 55 a mas')
            sys.exit(0)
        else:
            '''
            probable menor de 55
            '''
            pass
    else:
        '''
        Seguimos en los mismos parametros
        '''
        pass

print('todavia puedo calcular esos parametros espera actualizacion')
print(puntuacion)
