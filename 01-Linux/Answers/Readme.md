# Respuestas laboratorio 1

1. El comando grep busca dentro de un o varios archivos las lineas que concuerden con un patron especificado y por defecto las subraya. Por ejemplo

	``$grep "palabra" FILE_PATTERN``	Busca una palabra en los archivos que cumplan el patron.
	
	``$ grep -r "palabra" *``		Busca una palabra en todos los archivos de un directorio incluyendo los subdirectorios.



2. Al comienzo de cada script al poner el comando ``#! /bin/bash`` se le indica al computador en que interpretador de comandos leer el archivo, bash es el shell por defecto en linux.

3. Para saber cuantos usuarios estan conectados al servidor del curso se puede utilizar el comando ``who``, este nos indica el nombre de usuario, su terminal, el tiempo que han estado conectados y el nombre del host. (https://www.hscripts.com/es/tutoriales/linux-commands/who.html)

4. Una posible forma de saber los usuarios que existen en el servidor y su shell es obtenerla del directorio /etc/passwd en donde se encuentra toda la información de los usuarios. El elemento 1 y 7 de esta lista indican el nombre de usuario y el shell, es por esto que con el comando cut le decimos que solo tome esa informacion, luego los ordenamos con sort y con grep le decimos que muestre unicamente los usuarios que tienen como shell /bin/bash 

Comando - ``cut -f 1,7 -d ':' --output-delimiter=' ' /etc/passwd | sort | grep /bin/bash`` (https://www.computerhope.com/unix/ucut.htm)

Respuesta

	``acastillo /bin/bash
	alejandromateo /bin/bash
	diegoangulo /bin/bash
	estebanfederico /bin/bash
	ingcg /bin/bash
	jcleon /bin/bash
	juansantiago /bin/bash
	lapardo /bin/bash
	pa.arbelaez /bin/bash
	root /bin/bash
	rvandres /bin/bash
	vision /bin/bash``


5. Para encontrar imagenes duplicadas de utilizó el siguiente script

	``` #!/bin/bash

	cd .

	rep=$(cksum * | cut -f 1 -d ' ' | uniq -d)

	for nom in ${rep[*]}
	do

	   cksum * | cut -f 1,3 -d ' ' | grep $nom | cut -f 2 -d ' ' >> repetidas.txt

	echo $? estan repetidas

	done ```

	El comando ``cksum`` nos permite saber cuales imagenes son iguales debido al contenido ya que este numero es diferente al de cualquier otro archivo si su contenido es diferente. Los nombres de los archivos repetidos se guardan en un archivo de texto llamado repetidas.txt

6. Se descargó la base de datos bsds500 usando el comando "wget" con el link de descarga y se descomprimió usando el comando "tar".

7.El tamaño total de la carpeta BSR es de 12K. Esto se obtuvo con el comando "ls -lh".  

Utilizando el comando "ls -1 | wc -l" en cada una de las 3 carpetas que contienen las imágenes, se obtuvo un total de 300 imágenes, distribuidas de la siguiente manera:

	test: 200 imágenes
	train: 200 imágenes
	val: 100 imágenes 

8. Todas las imágenes están en formato JPG y su resolución es 321x481 o 481x321. Para las resoluciones, se utilizó el comando "identify * |grep -o "[[:digit:]]*x[[:digit:]]*", el cual da una lista con las dimansiones de cada imagen. (http://www.upubuntu.com/2012/02/how-to-check-image-resolution-using.html).

9. Para encontrar las imágenes en sentido landscape, se utilizó el comando find . -name '*.jpg' -exec file {} \; | sed 's/\(.*jpg\): .* \([0-9]* x [0-9]*\).*/\2 \1/' | awk 'int($1) < 400 {print}' (https://unix.stackexchange.com/questions/52906/find-images-by-size-find-file-awk).
	
	En este, se da la lista de imágenes con altura menor a 400, sabiendo que en este caso serán las que están en sentido landscape. 

10. Para cambiar el tamaño de las imágenes, se utilizó el siguiente comando: "for file in *.jpg; do convert -resize 256x256 -- "$file" "${file%%.jpg}-resized.jpg"; done", con el cual se cambian los tamaños de las imágenes a 256x256. (https://stackoverflow.com/questions/561895/resize-a-list-of-images-in-line-command)
