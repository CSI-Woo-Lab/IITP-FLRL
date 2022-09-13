app = ['single-detection', 'multi-detection', 'tracking']
scen = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad']
videoN = [7, 4, 5, 9, 15, 4, 12, 4]
label = ['Bicyclist', 'Pedestrian']

app_idx = 0 

print('Annotation/Label Convert Working,', app[app_idx])




x_max = 640
y_max = 640

for k in range(0,len(scen)):
	for kk in range(0,videoN[k]):
		o_path = 'origin/' + scen[k] + '/video' + str(kk) + '/annotations.txt'

		fr = open(o_path, 'r')
		lines = fr.readlines()

		base = list ()

		for line in lines:
			line = line.replace('\n', "")
			line = line.replace('\"', "")
			base.append(line.split(' '))

		fr.close ()


		for kkk in range(0,len(base)):
			frame = base[kkk][5]
			w_path = app[app_idx] + '/' + 'labels/' + scen[k] + '_' + frame + '.txt'

			fw = open(w_path, 'a')

			data =  

			fw.close ()
