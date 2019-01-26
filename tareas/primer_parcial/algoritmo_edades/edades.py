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
# sexo = input('Introduce tu sexo de nacimiento M/H:')
# if sexo == 'h':
#     estatura = 1.60
# else:
#     estatura = 1.50
# estatura_usuario = input('Introduce tu estatura: ')
# if float(estatura_usuario) > estatura:
#      puntuacion += 1
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
print('Casado, Soltero, Viudo, divorsiado')
estado_civil = input('C/S/V/D:')
if puntuacion < 4:
    '''
    Posiblemente adulto mayor
    '''
    nietos = input('Tiene nietos S/N: ')

    if nietos == 'S':
        '''
        Su edad minima es de 30 años
        '''
        if estado_civil == 'S':
            '''
            parametros extraños pero sigue siendo un adulto
            '''
            print('Calculo que tiene unso 45 a 60')
            sys.exit(0)
        else:
            '''
            Muy probable que sea adulto mayor
            '''
            print('Por sus datos calculo que su edad es 60 o Mas ')
            sys.exit(0)
    else:
        '''
        Probablemente un niño o adolecente
        '''
        casa = input('Vives en la casa de tus padres Y/N: ')
        if casa == 'Y':
            if puntuacion < 2:
                print('Por los datos que me diste calculo que eres un niño de entre 6 a 10')
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
        else:
            '''
            Probablemente un niño o adolecente
            '''
            horfanato = input('Vives en una horfanato propia S/N:').upper()
            if horfanato == 'S':
                '''
                Es un niño o adolecente
                '''
                if puntuacion < 2:
                    print('Por los datos que me diste calculo que eres un niño de entre 6 a 10')
                    sys.exit(0)
                else:
                    print('Por los datos que me diste calculo que tienes entre 10 y 15 años')
                    sys.exit(0)
            else:
                '''
                Probablemente un adulto o joven
                '''
                dependes = input('Tienes algun padre o tutor del que dependas con dinero S/N: ').upper()
                if dependes == 'S':
                    print('Calculo que tienes una edad ente 10 a 15 años')
                    sys.exit(0)
                else:
                    '''
                    Debe ser mayor a 12 años
                    '''
                    print('Debes ser mayor de 12, pero todavia')
                    print('me faltan parametros para saber tu edad')
                    print('mas exactamente ')
elif Escuela >= 4 :
    '''
    Debe ser mayo de 14
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
            ayuda_monetaria = input('Tus padres te ayudan monetariamente con tu hijo S/N: ').upper()
            if ayuda_monetaria == 'S':
                '''
                Probablemente sea menor de 24
                '''
                print('Calculo que tu edad esta entre 24 y 15 ')
                print('pero me faltan algunos parametros para')
                print('saber mas exactamente')
            else:
                '''
                Mismos parametros
                '''
                print('cual es el nivel maximo de estudio de tus hijos')
                print('Kinder                            1')
                print('Primaria                          2')
                print('Secundaria                        3')
                print('Preparatoria                      4')
                print('Universidad                       5')
                print('Mas                               6')
                escuela_hijos = int(input('Selecciona una opcion: '))
                if escuela_hijos >= 4:
                    '''
                    Probablemente 40 o mas
                    '''
                    pass
                else:
                    '''
                    Probablemente entre 22 a 40
                    '''
                    pass
    else:
        '''
        Seguimos en los mismos parametros
        '''
        pass

print('todavia puedo calcular esos parametros espera actualizacion')
