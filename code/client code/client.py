from Tkinter import *
import socket
import pickle
import graphlab
from mpl_toolkits.basemap import Basemap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import time
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import math
import Tix
import os
import ttk


def myfunction(event):
    canvas.configure(scrollregion=canvas.bbox("all"))


def myfunction2(event):
    can_width=event.width
    canvas.itemconfig(can_frame,width=can_width)

def bar(pb):
    pb['value']=20
    root.update_idletasks()
    time.sleep(1)
    pb['value']=50
    root.update_idletasks()
    time.sleep(1)
    pb['value']=80
    root.update_idletasks()
    time.sleep(1)
    pb['value']=100

def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    pb=ttk.Progressbar(topframe,orient='horizontal',length=100,mode='determinate')
    pb.pack(expand=True, fill=BOTH, side=TOP)
    bar(pb)
    python = sys.executable
    os.execl(python, python, * sys.argv)


def donothing():
   filewin = Toplevel(root)
   button = Button(filewin, text="Do nothing button")
   button.pack()

def about():
   filewin = Toplevel(root)
   l = Label(filewin, text="SHUSHRUTA is a Health Information System. It provides complete information of many diseases. Your Version is 1.0",height=20)
   l.pack(expand=True,fill=BOTH)

def clean_frame():
	vap.set("Choose your Attribute")
	vas1.set("Choose month")
	vas11.set("Choose year")
	vas22.set("Choose State")
	vas2.set("Choose list or map")
	vat.set("Choose year")
	for widget in subframe.winfo_children():
       		widget.destroy()
	for widget in subframe2.winfo_children():
       		widget.destroy()
	for widget in subframe3.winfo_children():
       		widget.destroy()
	for widget in subframe4.winfo_children():
       		widget.destroy()
	for widget in subframe5.winfo_children():
       		widget.destroy()
	for widget in subframe6.winfo_children():
       		widget.destroy()
			

attr_list=[]
count=1

def check_col_exs(col):
	for c in city_cols:
		if(c==col):
			return True
	return False

def draw_map(cities):
	print cities
	plt.figure('District_Distribution')
	plt.suptitle('District_Distribution')
	plt.title("Red- Highest Density, Green- Medium Density, Yellow- Lowest Density")
	map = Basemap(projection='lcc', lat_0=21, lon_0=84,
    		resolution = 'h', area_thresh = 1000.0,
    		llcrnrlon=66, llcrnrlat=6,
    		urcrnrlon=98, urcrnrlat=36)
	map.readshapefile('shp/IND_adm2', name='districts', drawbounds=True)
	map.drawcoastlines()
	map.drawcountries(linewidth=2)
	map.drawmapboundary()

	district_names=[]
	
	colors=['yellow','green','red']
	for shape_dict in map.districts_info:
		district_names.append(shape_dict['NAME_2'])

	l = len(cities)
	ax = plt.gca() # get current axes instance
	cnt=0
	print l
	print int(l/3)
	print int((2*l)/3)
	for city in cities:
		seg = map.districts[district_names.index(city)]
		print cnt
		if cnt>=0 and cnt<int(l/3):
			poly = Polygon(seg, facecolor='yellow')
			ax.add_patch(poly)
			print 'yellow wala'

		elif cnt>=int(l/3) and cnt<int((2*l)/3):
			poly = Polygon(seg, facecolor='green')
			ax.add_patch(poly)
			print 'green wala'

		elif cnt>=int((2*l)/3):
			poly = Polygon(seg, facecolor='red')
			ax.add_patch(poly)
			print 'red wala'
		
		cnt=cnt+1	
			
	plt.show()

	
	

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

def draw_map2(states_rank):
	plt.figure('State_Distribution')
	plt.suptitle('State_Distribution')
	plt.title("Red- Highest Density, Green- Medium Density, Yellow- Lowest Density")
	map = Basemap(projection='lcc', lat_0=21, lon_0=84,
    		resolution = 'h', area_thresh = 1000.0,
    		llcrnrlon=66, llcrnrlat=6,
    		urcrnrlon=98, urcrnrlat=36)
	map.readshapefile('shp/IND_adm1', name='states', drawbounds=True)
	map.drawcoastlines()
	map.drawcountries(linewidth=2)
	map.drawmapboundary()
	colors=['yellow','green','red']
	state_names = []
	for shape_dict in map.states_info:
	    state_names.append(shape_dict['NAME_1'])
	
	ax = plt.gca() # get current axes instance
	cnt=0
	for state in states_rank:
		seg = map.states[state_names.index(state)]
		poly = Polygon(seg, facecolor=colors[cnt])
		cnt=cnt+1
		ax.add_patch(poly)

	plt.show()
	
def show_map1():
	dis=var.get()
	if(dis=='Heart-Disease'):
			s = socket.socket()
			host = socket.gethostname()
			port = 12345 
			s.connect((host, port))
        		print s.recv(1024)
			s.send('Load Map District')
			recv=s.recv(1024)
			distribution_list=pickle.loads(recv)
			print distribution_list
			draw_map(distribution_list)
			s.close 
				
	elif(dis=='Chronic Kidney Disease'):
                        s = socket.socket()
			host = socket.gethostname()
			port = 12347 
			s.connect((host, port))
        		print s.recv(1024)
			s.send('Load Map District')
			recv=s.recv(1024)
			distribution_list=pickle.loads(recv)
			print distribution_list
			draw_map(distribution_list)
			s.close 
			
	elif(dis=='Cardiotographic(CTG)'):
                        s = socket.socket()
			host = socket.gethostname()
			port = 12349
			s.connect((host, port))
        		print s.recv(1024)
			s.send('Load Map District')
			recv=s.recv(1024)
			distribution_list=pickle.loads(recv)
			print distribution_list
			draw_map(distribution_list)
			s.close 

	elif(dis=='Cervical-Cancer'):
                        s = socket.socket()
			host = socket.gethostname()
			port = 12351 
			s.connect((host, port))
        		print s.recv(1024)
			s.send('Load Map District')
			recv=s.recv(1024)
			distribution_list=pickle.loads(recv)
			print distribution_list
			draw_map(distribution_list)
			s.close 			

def show_map2():
	dis=var.get()
	if(dis=='Heart-Disease'):
			s = socket.socket()
			host = socket.gethostname()
			port = 12345 
			s.connect((host, port))
			print s.recv(1024)
			s.send('Load Map State')
			recv=s.recv(1024)
			distribution_list=pickle.loads(recv)
			print distribution_list
			draw_map2(distribution_list)
			s.close 


	elif(dis=='Chronic Kidney Disease'):
			s = socket.socket()
			host = socket.gethostname()
			port = 12347 
			s.connect((host, port))
			print s.recv(1024)
			s.send('Load Map State')
			recv=s.recv(1024)
			distribution_list=pickle.loads(recv)
			print distribution_list
			draw_map2(distribution_list)
			s.close 
	
	elif(dis=='Cardiotographic(CTG)'):
			s = socket.socket()
			host = socket.gethostname()
			port = 12349 
			s.connect((host, port))
			print s.recv(1024)
			s.send('Load Map State')
			recv=s.recv(1024)
			distribution_list=pickle.loads(recv)
			print distribution_list
			draw_map2(distribution_list)
			s.close 

	elif(dis=='Cervical-Cancer'):
			s = socket.socket()
			host = socket.gethostname()
			port = 12351 
			s.connect((host, port))
			print s.recv(1024)
			s.send('Load Map State')
			recv=s.recv(1024)
			distribution_list=pickle.loads(recv)
			print distribution_list
			draw_map2(distribution_list)
			s.close 

		
def send():
    dis=var.get()
    vals={}

    if dis=='Heart-Disease':
		cnt=0
		attrs=['metformin','repaglinide','glimepiride','glipizide','glyburide','pioglitazone',
			'rosiglitazone','change']
		data_types=['str','str','str','str','str','str','str','str']
		for v in attr_list:
			temp=v.get()
			if(temp==attrs[cnt]):
				toplevel = Toplevel()
                                label1 = Label(toplevel, text="Please Select A Valid Value", height=50, width=100)
                                label1.pack()
                                return
			if(data_types[cnt]=='str'):
				vals[attrs[cnt]]=temp
			elif(data_types[cnt]=='flt'):
				vals[attrs[cnt]]=float(temp)
			elif(data_types[cnt]=='int'):
				vals[attrs[cnt]]=int(temp)
			cnt=cnt+1
	        print vals
		s = socket.socket()
		host = socket.gethostname()
		port = 12345 
		s.connect((host, port))
		print s.recv(1024)
		data=pickle.dumps(vals)
		s.send(data)
		result=s.recv(1024)
		toplevel = Toplevel()
		if(result=="Yes"):
			la=Label(toplevel,text="Results are positive", height=50, width=100)
			la.pack()
		else:
			la=Label(toplevel,text="Results are negative", height=50, width=100)
			la.pack()
                s.close

    elif dis=='Cardiotographic(CTG)':
                cnt=0
		attrs=['year','DP','DS.1','Tendency','CLASS']
		data_types=['int','int','flt','int','int']
		for v in attr_list:
			temp=v.get()
			if(temp==attrs[cnt]):
				toplevel = Toplevel()
                                label1 = Label(toplevel, text="Please Select A Valid Value", height=50, width=100)
                                label1.pack()
                                return
			if(data_types[cnt]=='str'):
				vals[attrs[cnt]]=temp
			elif(data_types[cnt]=='flt'):
				vals[attrs[cnt]]=float(temp)
			elif(data_types[cnt]=='int'):
				vals[attrs[cnt]]=int(temp)
			cnt=cnt+1
		print vals
		s = socket.socket()
	        host = socket.gethostname()
		port = 12349 
		s.connect((host, port))
		print s.recv(1024)
		data=pickle.dumps(vals)
		s.send(data)
		result=s.recv(1024)
		toplevel = Toplevel()
		if(result=="1"):
			la=Label(toplevel,text="Results Are Normal", height=50, width=100)
			la.pack()
		elif(result=="2"):
			la=Label(toplevel,text="Biopsy Is Suspected", height=50, width=100)
			la.pack()
		else:
			ls=Label(toplevel,text="Results Are Pathologic", height=50, width=100) 
			ls.pack()  
                s.close 
		
    elif dis == 'Chronic Kidney Disease' :
		cnt=0
		attrs=['rbc','pc','pcv','htn','dm','appet','pe','ane']
		data_types=['str','str','int','str','str','str','str','str']
		for v in attr_list:
			if(attrs[cnt]=='pcv'):
				tt=v.get()
				ll=[x.strip() for x in tt.split('-')]
				ss=(int(ll[0])+int(1))/2
				v.set(ss)
			temp=v.get()
			if(temp==attrs[cnt]):
				toplevel = Toplevel()
                                label1 = Label(toplevel, text="Please Select A Valid Value", height=50, width=100)
                                label1.pack()
                                return
			if(data_types[cnt]=='str'):
				vals[attrs[cnt]]=temp
			elif(data_types[cnt]=='flt'):
				vals[attrs[cnt]]=float(temp)
			elif(data_types[cnt]=='int'):
				vals[attrs[cnt]]=int(temp)
			cnt=cnt+1
	        print vals
		s = socket.socket()
		host = socket.gethostname()
		port = 12347 
		s.connect((host, port))
		print s.recv(1024)
		data=pickle.dumps(vals)
		s.send(data)
		result=s.recv(1024)
		toplevel = Toplevel()
		if(result=="Yes"):
			la=Label(toplevel,text="Results are positive", height=50, width=100)
			la.pack()
		else:
			la=Label(toplevel,text="Results are negative", height=50, width=100)
			la.pack()
                s.close 
 
    elif dis=='Cervical-Cancer':
		cnt=0
		attrs=['year','Smokes','Hormonal Contraceptives (years)','STDs:genital herpes','STDs:HIV','Dx:CIN','Dx:HPV',
				'Schiller']
		data_types=['int','int','flt','int','int','int','int','int']
		for v in attr_list:
			if(attrs[cnt]=='Hormonal Contraceptives (years)'):
				tt=v.get()
				ll=[x.strip() for x in tt.split('-')]
				ss=(float(ll[0])+float(1))/2
				v.set(ss)
			temp=v.get()
			if(temp==attrs[cnt]):
				toplevel = Toplevel()
                                label1 = Label(toplevel, text="Please Select A Valid Value", height=50, width=100)
                                label1.pack()
                                return
			if(data_types[cnt]=='str'):
				vals[attrs[cnt]]=temp
			elif(data_types[cnt]=='flt'):
				vals[attrs[cnt]]=float(temp)
			elif(data_types[cnt]=='int'):
				vals[attrs[cnt]]=int(temp)
			cnt=cnt+1
		print vals
		s = socket.socket()
		host = socket.gethostname()
		port = 12351
		s.connect((host, port))
		print s.recv(1024)
		data=pickle.dumps(vals)
		s.send(data)
		result=s.recv(1024)
		toplevel = Toplevel()
		if(result=="Yes"):
			la=Label(toplevel,text="Biopsy is 1", height=50, width=100)
			la.pack()
		else:
			la=Label(toplevel,text="Biopsy is 0", height=50, width=100)
			la.pack()
                s.close 

def col_dist():
        dis=var.get()
	if dis=='Heart-Disease' :
		t=vap.get()
		if(t=="Choose your Attribute"):
			toplevel = Toplevel()
			label1 = Label(toplevel, text="Please Select A Valid Attribute", height=50, width=100)
			label1.pack()
			return
		s = socket.socket()
		host = socket.gethostname()
		port = 12345 
		s.connect((host, port))
		print s.recv(1024)
		s.send('column distribution')
		msg=s.recv(1024)
		if(msg=='ok'):
			s.send(t)
			plt.figure(t)
			plt.suptitle(t)
			rec=s.recv(4096)
			[y,x,l]=pickle.loads(rec)
			r=range(l)
			plt.bar(r,y,align='center')
			plt.xticks(r,x)
			plt.show()
                s.close 

	elif dis=='Cervical-Cancer' :
		t=vap.get()
		if(t=="Choose your Attribute"):
			toplevel = Toplevel()
			label1 = Label(toplevel, text="Please Select A Valid Attribute", height=50, width=100)
			label1.pack()
			return
		s = socket.socket()
		host = socket.gethostname()
		port = 12351 
		s.connect((host, port))
		print s.recv(1024)
		s.send('column distribution')
		msg=s.recv(1024)
		if(msg=='ok'):
			s.send(t)
			plt.figure(t)
			plt.suptitle(t)
			rec=s.recv(4096)
			[y,x,l]=pickle.loads(rec)
			r=range(l)
			plt.bar(r,y,align='center')
			plt.xticks(r,x)
			plt.show()
                s.close 

	elif dis=='Cardiotographic(CTG)' :
		t=vap.get()
		if(t=="Choose your Attribute"):
			toplevel = Toplevel()
			label1 = Label(toplevel, text="Please Select A Valid Attribute", height=50, width=100)
			label1.pack()
			return
		s = socket.socket()
		host = socket.gethostname()
		port = 12349 
		s.connect((host, port))
		print s.recv(1024)
		s.send('column distribution')
		msg=s.recv(1024)
		if(msg=='ok'):
			s.send(t)
			plt.figure(t)
			plt.suptitle(t)
			rec=s.recv(4096)
			[y,x,l]=pickle.loads(rec)
			r=range(l)
			plt.bar(r,y,align='center')
			plt.xticks(r,x)
			plt.show()
                s.close 

	elif dis=='Chronic Kidney Disease' :
		print "here"
		t=vap.get()
		if(t=="Choose your Attribute"):
			toplevel = Toplevel()
			label1 = Label(toplevel, text="Please Select A Valid Attribute", height=50, width=100)
			label1.pack()
			return
		s = socket.socket()
		host = socket.gethostname()
		port = 12347 
		s.connect((host, port))
		print s.recv(1024)
		s.send('column distribution')
		msg=s.recv(1024)
		if(msg=='ok'):
			s.send(t)
			plt.figure(t)
			plt.suptitle(t)
			rec=s.recv(4096)
			[y,x,l]=pickle.loads(rec)
			r=range(l)
			plt.bar(r,y,align='center')
			plt.xticks(r,x)
            		plt.show()
                s.close 
		
			
def curr_areas():
	dis=var.get()
	if(dis=='Heart-Disease'):
		mm=vas1.get()
		yy=vas11.get()
		ss=vas22.get()
		op=vas2.get()

		if(mm=="Choose month" or yy=="Choose year" or ss=="Choose State" or op=="Choose list or map"):
			toplevel = Toplevel()
	    		label1 = Label(toplevel, text="Please Select A Valid Value", height=0, width=20)
	    		label1.pack()
		else:
			s = socket.socket()
			host = socket.gethostname()
			port = 12345 
			s.connect((host, port))
			print s.recv(1024)
			s.send('currently affected areas')
			msg=s.recv(4096)
			if(msg=='ok'):
				s.send(mm)
				msg2=s.recv(4096)
				if(msg2=='ok'):
					s.send(yy)
					msg3=s.recv(4096)
					if(msg3=='ok'):
						s.send(ss)
						msg4=s.recv(4096)
						X=pickle.loads(msg4)
						if(str(X)=="No Area Is Currently Affected"):
							toplevel=Toplevel()
							l= Label(toplevel, text="No Area Is Currently Affected", height=0, width=20)
							l.pack()
							return
						if(op=='Map'):
							draw_map(X)
						else :	
							toplevel = Toplevel()
							for dis in X:
									l= Label(toplevel, text=str(dis), height=0, width=20)
									l.pack()
                        s.close 

	elif(dis=='Cervical-Cancer'):
		mm=vas1.get()
		yy=vas11.get()
		ss=vas22.get()
		op=vas2.get()

		if(mm=="Choose month" or yy=="Choose year" or ss=="Choose State" or op=="Choose list or map"):
			toplevel = Toplevel()
	    		label1 = Label(toplevel, text="Please Select A Valid Value", height=0, width=20)
	    		label1.pack()
		else:
			s = socket.socket()
			host = socket.gethostname()
			port = 12351 
			s.connect((host, port))
			print s.recv(1024)
			s.send('currently affected areas')
			msg=s.recv(4096)
			if(msg=='ok'):
				s.send(mm)
				msg2=s.recv(4096)
				if(msg2=='ok'):
					s.send(yy)
					msg3=s.recv(4096)
					if(msg3=='ok'):
						s.send(ss)
						msg4=s.recv(4096)
						X=pickle.loads(msg4)
						if(str(X)=="No Area Is Currently Affected"):
							toplevel=Toplevel()
							l= Label(toplevel, text="No Area Is Currently Affected", height=0, width=20)
							l.pack()
							return
						if(op=='Map'):
							draw_map(X)
						else :	
							toplevel = Toplevel()
							for dis in X:
									l= Label(toplevel, text=str(dis), height=0, width=20)
									l.pack()
			s.close 


	elif(dis=='Cardiotographic(CTG)'):
		mm=vas1.get()
		yy=vas11.get()
		ss=vas22.get()
		op=vas2.get()

		if(mm=="Choose month" or yy=="Choose year" or ss=="Choose State" or op=="Choose list or map"):
			toplevel = Toplevel()
	    		label1 = Label(toplevel, text="Please Select A Valid Value", height=0, width=20)
	    		label1.pack()
		else:
			s = socket.socket()
			host = socket.gethostname()
			port = 12349 
			s.connect((host, port))
			print s.recv(1024)
			s.send('currently affected areas')
			msg=s.recv(4096)
			if(msg=='ok'):
				s.send(mm)
				msg2=s.recv(4096)
				if(msg2=='ok'):
					s.send(yy)
					msg3=s.recv(4096)
					if(msg3=='ok'):
						s.send(ss)
						msg4=s.recv(4096)
						X=pickle.loads(msg4)
						if(str(X)=="No Area Is Currently Affected"):
							toplevel=Toplevel()
							l= Label(toplevel, text="No Area Is Currently Affected", height=0, width=20)
							l.pack()
							return
						if(op=='Map'):
							draw_map(X)
						else :	
							toplevel = Toplevel()
							for dis in X:
									l= Label(toplevel, text=str(dis), height=0, width=20)
									l.pack()
			s.close 

	elif(dis=='Chronic Kidney Disease'):
		mm=vas1.get()
		yy=vas11.get()
		ss=vas22.get()
		op=vas2.get()

		if(mm=="Choose month" or yy=="Choose year" or ss=="Choose State" or op=="Choose list or map"):
			toplevel = Toplevel()
	    		label1 = Label(toplevel, text="Please Select A Valid Value", height=0, width=20)
	    		label1.pack()
		else:
			s = socket.socket()
			host = socket.gethostname()
			port = 12347
			s.connect((host, port))
			print s.recv(1024)
			s.send('currently affected areas')
			msg=s.recv(4096)
			if(msg=='ok'):
				s.send(mm)
				msg2=s.recv(4096)
				if(msg2=='ok'):
					s.send(yy)
					msg3=s.recv(4096)
					if(msg3=='ok'):
						s.send(ss)
						msg4=s.recv(4096)
						X=pickle.loads(msg4)
						if(str(X)=="No Area Is Currently Affected"):
							toplevel=Toplevel()
							l= Label(toplevel, text="No Area Is Currently Affected", height=0, width=20)
							l.pack()
							return
						if(op=='Map'):
							draw_map(X)
						else :	
							toplevel = Toplevel()
							for dis in X:
									l= Label(toplevel, text=str(dis), height=0, width=20)
									l.pack()
                        s.close 


	
		
def epicenter():
	dis=var.get()
	if(dis=='Heart-Disease'):
		s = socket.socket()
		host = socket.gethostname()
		port = 12345 
		s.connect((host, port))
        	print s.recv(1024)
		s.send('starting location')
		msg=s.recv(4096)
		if(str(msg)=="This Disease Is Not Found Yet Anywhere"):
			toplevel = Toplevel()
    			label1 = Label(toplevel, text="This Disease Is Not Found Yet Anywhere", height=50, width=100)
    			label1.pack()
			return
		else:
			plt.figure("Starting Locations")
			plt.suptitle("Starting Locations")
			[y,x,l]=pickle.loads(msg)
			print y
			print x
			r=range(l)
        		plt.bar(r,y,align='center')
			plt.xticks(r,x)
			plt.show()
                s.close 

	elif(dis=='Cervical-Cancer'):
		s = socket.socket()
		host = socket.gethostname()
		port = 12351
		s.connect((host, port))
        	print s.recv(1024)
		s.send('starting location')
		msg=s.recv(4096)
		if(str(msg)=="This Disease Is Not Found Yet Anywhere"):
			toplevel = Toplevel()
    			label1 = Label(toplevel, text="This Disease Is Not Found Yet Anywhere", height=50, width=100)
    			label1.pack()
			return
		else:
			plt.figure("Starting Locations")
			plt.suptitle("Starting Locations")
			[y,x,l]=pickle.loads(msg)
			r=range(l)
        		plt.bar(r,y,align='center')
			plt.xticks(r,x)
			plt.show()
                s.close 


	elif(dis=='Cardiotographic(CTG)'):
		s = socket.socket()
		host = socket.gethostname()
		port = 12349
		s.connect((host, port))
        	print s.recv(1024)
		s.send('starting location')
		msg=s.recv(4096)
		if(str(msg)=="This Disease Is Not Found Yet Anywhere"):
			toplevel = Toplevel()
    			label1 = Label(toplevel, text="This Disease Is Not Found Yet Anywhere", height=50, width=100)
    			label1.pack()
			return
		else:
			plt.figure("Starting Locations")
			plt.suptitle("Starting Locations")
			[y,x,l]=pickle.loads(msg)
			r=range(l)
        		plt.bar(r,y,align='center')
			plt.xticks(r,x)
            		plt.show()
                s.close 

	elif(dis=='Chronic Kidney Disease'):
		s = socket.socket()
		host = socket.gethostname()
		port = 12347
		s.connect((host, port))
        	print s.recv(1024)
		s.send('starting location')
		msg=s.recv(4096)
		if(str(msg)=="This Disease Is Not Found Yet Anywhere"):
			toplevel = Toplevel()
    			label1 = Label(toplevel, text="This Disease Is Not Found Yet Anywhere", height=50, width=100)
    			label1.pack()
			return
		else:
			plt.figure("Starting Locations")
			plt.suptitle("Starting Locations")
			[y,x,l]=pickle.loads(msg)
			r=range(l)
        		plt.bar(r,y,align='center')
			plt.xticks(r,x)
			plt.show()
			
                s.close 


def year_wise():
	dis=var.get()
	if dis=='Heart-Disease' :
		t='year'
		s = socket.socket()
		host = socket.gethostname()
		port = 12345 
		s.connect((host, port))
		print s.recv(1024)
		s.send('column distribution')
		msg=s.recv(1024)
		if(msg=='ok'):
			s.send(t)
			plt.figure(t)
			plt.suptitle(t)
			rec=s.recv(4096)
			[y,x,l]=pickle.loads(rec)
			r=range(l)
			plt.bar(r,y,align='center')
			plt.xticks(r,x)
			plt.show()
                s.close 

	elif dis=='Cervical-Cancer' :
		t='year'
		s = socket.socket()
		host = socket.gethostname()
		port = 12351 
		s.connect((host, port))
		print s.recv(1024)
		s.send('column distribution')
		msg=s.recv(1024)
		if(msg=='ok'):
			s.send(t)
			plt.figure(t)
			plt.suptitle(t)
			rec=s.recv(4096)
			[y,x,l]=pickle.loads(rec)
			r=range(l)
			plt.bar(r,y,align='center')
			plt.xticks(r,x)
			plt.show()
                s.close 

	elif dis=='Cardiotographic(CTG)' :
		t='year'
		s = socket.socket()
		host = socket.gethostname()
		port = 12349 
		s.connect((host, port))
		print s.recv(1024)
		s.send('column distribution')
		msg=s.recv(1024)
		if(msg=='ok'):
			s.send(t)
			plt.figure(t)
			plt.suptitle(t)
			rec=s.recv(4096)
			[y,x,l]=pickle.loads(rec)
			r=range(l)
			plt.bar(r,y,align='center')
			plt.xticks(r,x)
			plt.show()
                s.close 

	elif dis=='Chronic Kidney Disease' :
		t='year'
		s = socket.socket()
		host = socket.gethostname()
		port = 12347 
		s.connect((host, port))
		print s.recv(1024)
		s.send('column distribution')
		msg=s.recv(1024)
		if(msg=='ok'):
			s.send(t)
			plt.figure(t)
			plt.suptitle(t)
			rec=s.recv(4096)
			[y,x,l]=pickle.loads(rec)
			r=range(l)
			plt.bar(r,y,align='center')
			plt.xticks(r,x)
            		plt.show()
                s.close 
		

def month_wise():
	dis=var.get()
	if(dis=='Heart-Disease'):
		yy=vat.get()

		if(yy=="Choose year"):
			toplevel = Toplevel()
	    		label1 = Label(toplevel, text="Please Select A Valid Value", height=50, width=100)
	    		label1.pack()
			return
		else:
			s = socket.socket()
			host = socket.gethostname()
			port = 12345 
			s.connect((host, port))
			print s.recv(1024)
			s.send('timeline_year')
			msg=s.recv(4096)
			if(msg=='ok'):
				print "Year Is"
				print yy
				s.send(yy)
				msg2=s.recv(4096)
				plt.figure("Month Wise Distribution for year " + yy)
				plt.suptitle("Month Wise Distribution for year " + yy)
				[y,x,l]=pickle.loads(msg2)
				r=range(l)
        			plt.bar(r,y,align='center')
				plt.xticks(r,x)
				plt.show()
	        s.close 
					
	
	elif(dis=='Cervical-Cancer'):
		yy=vat.get()

		if(yy=="Choose year"):
			toplevel = Toplevel()
	    		label1 = Label(toplevel, text="Please Select A Valid Value", height=50, width=100)
	    		label1.pack()
			return 
		else:
			s = socket.socket()
			host = socket.gethostname()
			port = 12351 
			s.connect((host, port))
			print s.recv(1024)
			s.send('timeline_year')
			msg=s.recv(4096)
			if(msg=='ok'):
				s.send(yy)
				msg2=s.recv(4096)
				plt.figure("Month Wise Distribution for year " + yy)
				plt.suptitle("Month Wise Distribution for year " + yy)
				[y,x,l]=pickle.loads(msg2)
				r=range(l)
        			plt.bar(r,y,align='center')
				plt.xticks(r,x)
				plt.show()
			s.close 
			


	elif(dis=='Cardiotographic(CTG)'):
		yy=vat.get()

		if(yy=="Choose year"):
			toplevel = Toplevel()
	    		label1 = Label(toplevel, text="Please Select A Valid Value", height=50, width=100)
	    		label1.pack()
			return 
		else:
			s = socket.socket()
			host = socket.gethostname()
			port = 12349 
			s.connect((host, port))
			print s.recv(1024)
			s.send('timeline_year')
			msg=s.recv(4096)
			if(msg=='ok'):
				s.send(yy)
				msg2=s.recv(4096)
				plt.figure("Month Wise Distribution for year " + yy)
				plt.suptitle("Month Wise Distribution for year " + yy)
				[y,x,l]=pickle.loads(msg2)
				r=range(l)
        			plt.bar(r,y,align='center')
				plt.xticks(r,x)
				plt.show()
                        s.close 		

	elif(dis=='Chronic Kidney Disease'):
		yy=vat.get()

		if(yy=="Choose year"):
			toplevel = Toplevel()
	    		label1 = Label(toplevel, text="Please Select A Valid Value", height=50, width=100)
	    		label1.pack()
			return
		else:
			s = socket.socket()
			host = socket.gethostname()
			port = 12347 
			s.connect((host, port))
			print s.recv(1024)
			s.send('timeline_year')
			msg=s.recv(4096)
			if(msg=='ok'):
				s.send(yy)
				msg2=s.recv(4096)
				plt.figure("Month Wise Distribution for year " + yy)
				plt.suptitle("Month Wise Distribution for year " + yy)
				[y,x,l]=pickle.loads(msg2)
				r=range(l)
        			plt.bar(r,y,align='center')
				plt.xticks(r,x)
				plt.show()
                        s.close 

def select():
	count=0
	dis=var.get()
	print "You Selected "+dis
	clean_frame()
	
	if(dis=='Choose your Disease'):
		toplevel = Toplevel()
	    	label1 = Label(toplevel, text="First Choose A Valid Disease", height=50, width=100)
	    	label1.pack()
		return 

	b = Label(subframe, text="Enter your Details",font="Verdana 15 bold", bg="orange", fg="black")
	b.pack(padx=5, pady=10, anchor='w')
	del attr_list[:]
	if(dis=='Heart-Disease'):
		attrs=['metformin','repaglinide','glimepiride','glipizide','glyburide','pioglitazone',
			'rosiglitazone','change']
		attr_vals=[['Steady','No','Down','Up'],['Steady','No'],['Steady','No']
				,['Steady','No'],['Steady','No','Down','Up'],['Steady','No'],['Steady','No'],['Ch','No']]
		
		ch=['state','district','race','gender','payer_code','medical_specialty','metformin','repaglinide',
		'nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone',
		'acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin',
		'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','change','readmitted']
		
		for attr in attrs:		
				va = StringVar(root)
				va.set(attr)
				choices=attr_vals[count]
				option = OptionMenu(subframe, va, *choices)
				option.configure(font="Verdana 10 bold",fg="black")
				option.pack(padx='10',side='left',pady='5')
				attr_list.append(va)
				count=count+1
				
		bt1=Button(subframe,text="Submit",fg = "red",font="Verdana 8 bold",command=send)
		bt1.pack(anchor="center", pady='10')	

	#Geographica_Information_System
		
		w = Label(subframe2, text="njkengjs", bg="black", fg="black")
		w.pack(fill=X,anchor="center",pady='20')
		y = Label(subframe2, text="Geographical Information System",font="Verdana 15 bold", bg="orange", fg="black")
		y.pack(padx=5, pady=5, anchor='w')
		bt2=Button(subframe2,text="District-Wise Distribution",fg = "black",font="Verdana 10 bold",command=show_map1)
		bt2.pack(side='left',padx=10,pady=10)
		bt3=Button(subframe2,text="State-Wise Distribution",fg = "black",font="Verdana 10 bold",command=show_map2)
		bt3.pack(side='left',padx=10,pady=10)	
		bt4=Button(subframe2,text="Epicenter",fg = "black",font="Verdana 10 bold",command=epicenter)
		bt4.pack(side='left',padx=10,pady=10)	
		
	#Attribute_Wise_Distribution

		z = Label(subframe3, text="njkengjs", bg="black", fg="black")
		z.pack(fill=X,anchor="center",pady='20')
		a = Label(subframe3, text="Attribute based Distribution",font="Verdana 15 bold", bg="orange", fg="black")
		a.pack(padx=5, pady=5, anchor='w')
		option = OptionMenu(subframe3, vap, *ch)
		option.pack(side='left',padx=10,pady=15)
		option.configure(font="Verdana 10 bold",fg="black")
		bt5=Button(subframe3,text="Submit",fg = "red",font="Verdana 8 bold",command=col_dist)
		bt5.pack(side='left',padx='15',pady=15)	
		
		
	#Timeline

		c = Label(subframe4, text="njkengjs", bg="black", fg="black")
		c.pack(fill=X,anchor="center",pady='20')
		d = Label(subframe4, text="Timeline",font="Verdana 15 bold", bg="orange", fg="black")
		d.pack(padx=5, pady=5, anchor='w')
		
		#Location_affected_at_specific_time

		e = Label(subframe4, text=">> Location affected at specific Time",font="Verdana 12 bold", bg="orange", fg="black")
		e.pack(padx=5, pady=5, anchor='w')
		option1 = OptionMenu(subframe4, vas1, *ch_month)
		option1.pack(side='left',padx=10,pady=10)
		option1.configure(font="Verdana 10 bold",fg="black")
		option11 = OptionMenu(subframe4, vas11, *ch_year)
		option11.pack(side='left',padx=10,pady=10)
		option11.configure(font="Verdana 10 bold",fg="black")
		option22 = OptionMenu(subframe4, vas22, *ch_state)
		option22.pack(side='left',padx=10,pady=10)
		option22.configure(font="Verdana 10 bold",fg="black")
		option2 = OptionMenu(subframe4, vas2, *ch_form)
		option2.pack(side='left',padx=10,pady=10)
		option2.configure(font="Verdana 10 bold",fg="black")
		bt6=Button(subframe4,text="Submit",fg = "red",font="Verdana 8 bold",command=curr_areas)
		bt6.pack(side='left',padx='15',pady=10)
		
		#Time_wise_distribution
		
		f = Label(subframe5, text=">> Time-wise Distribution",font="Verdana 12 bold", bg="orange", fg="black")
		f.pack(padx=5, pady=5, anchor='w')	
		bt6=Button(subframe5,text="Year-wise Distribution",fg = "black",font="Verdana 10 bold",command=year_wise)
		bt6.pack(side='left',padx='20',pady=10)
		g = Label(subframe5, text="Month-wise Distribution ->",font="Verdana 10 bold", bg="orange", fg="Red")
		g.pack(padx=20, pady=10, side='left')	
		option3 = OptionMenu(subframe5, vat, *ch_year)
		option3.pack(side='left',padx=10,pady=10)
		option3.configure(font="Verdana 10 bold",fg="black")
		bt7=Button(subframe5,text="OK",fg = "red",font="Verdana 8 bold",command=month_wise)
		bt7.pack(side='left',padx='10',pady=10)
		
	
	elif(dis=='Chronic Kidney Disease'):
		attrs=['rbc','pc','pcv','htn','dm','appet','pe','ane']
		attr_vals=[['normal','abnormal'],['normal','abnormal'],['20-30','30-40','40-50','50-60'],['yes','no'],['yes','no'],['good','poor'],['yes','no'],['yes','no']]
		
		ch=['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']
		
		for attr in attrs:		
				va = StringVar(root)
				va.set(attr)
				choices=attr_vals[count]
				option = OptionMenu(subframe, va, *choices)
				option.configure(font="Verdana 10 bold",fg="black")
				option.pack(padx='10',side='left',pady='5')
				attr_list.append(va)
				count=count+1
		
		bt1=Button(subframe,text="Submit",fg = "red",font="Verdana 8 bold",command=send)
		bt1.pack(anchor="center", pady='10')	

	#Geographica_Information_System
		
		w = Label(subframe2, text="njkengjs", bg="black", fg="black")
		w.pack(fill=X,anchor="center",pady='20')
		y = Label(subframe2, text="Geographical Information System",font="Verdana 15 bold", bg="orange", fg="black")
		y.pack(padx=5, pady=5, anchor='w')
		bt2=Button(subframe2,text="District-Wise Distribution",fg = "black",font="Verdana 10 bold",command=show_map1)
		bt2.pack(side='left',padx=10,pady=10)
		bt3=Button(subframe2,text="State-Wise Distribution",fg = "black",font="Verdana 10 bold",command=show_map2)
		bt3.pack(side='left',padx=10,pady=10)	
		bt4=Button(subframe2,text="Epicenter",fg = "black",font="Verdana 10 bold",command=epicenter)
		bt4.pack(side='left',padx=10,pady=10)	
		
	#Attribute_Wise_Distribution

		z = Label(subframe3, text="njkengjs", bg="black", fg="black")
		z.pack(fill=X,anchor="center",pady='20')
		a = Label(subframe3, text="Attribute based Distribution",font="Verdana 15 bold", bg="orange", fg="black")
		a.pack(padx=5, pady=5, anchor='w')
		option = OptionMenu(subframe3, vap, *ch)
		option.pack(side='left',padx=10,pady=15)
		option.configure(font="Verdana 10 bold",fg="black")
		bt5=Button(subframe3,text="Submit",fg = "red",font="Verdana 8 bold",command=col_dist)
		bt5.pack(side='left',padx='15',pady=15)	
		
		
	#Timeline

		c = Label(subframe4, text="njkengjs", bg="black", fg="black")
		c.pack(fill=X,anchor="center",pady='20')
		d = Label(subframe4, text="Timeline",font="Verdana 15 bold", bg="orange", fg="black")
		d.pack(padx=5, pady=5, anchor='w')
		
		#Location_affected_at_specific_time

		e = Label(subframe4, text=">> Location affected at specific Time",font="Verdana 12 bold", bg="orange", fg="black")
		e.pack(padx=5, pady=5, anchor='w')
		option1 = OptionMenu(subframe4, vas1, *ch_month)
		option1.pack(side='left',padx=10,pady=10)
		option1.configure(font="Verdana 10 bold",fg="black")
		option11 = OptionMenu(subframe4, vas11, *ch_year)
		option11.pack(side='left',padx=10,pady=10)
		option11.configure(font="Verdana 10 bold",fg="black")
		option22 = OptionMenu(subframe4, vas22, *ch_state)
		option22.pack(side='left',padx=10,pady=10)
		option22.configure(font="Verdana 10 bold",fg="black")
		option2 = OptionMenu(subframe4, vas2, *ch_form)
		option2.pack(side='left',padx=10,pady=10)
		option2.configure(font="Verdana 10 bold",fg="black")
		bt6=Button(subframe4,text="Submit",fg = "red",font="Verdana 8 bold",command=curr_areas)
		bt6.pack(side='left',padx='15',pady=10)
		
		#Time_wise_distribution
		
		f = Label(subframe5, text=">> Time-wise Distribution",font="Verdana 12 bold", bg="orange", fg="black")
		f.pack(padx=5, pady=5, anchor='w')	
		bt6=Button(subframe5,text="Year-wise Distribution",fg = "black",font="Verdana 10 bold",command=year_wise)
		bt6.pack(side='left',padx='20',pady=10)
		g = Label(subframe5, text="Month-wise Distribution ->",font="Verdana 10 bold", bg="orange", fg="Red")
		g.pack(padx=20, pady=10, side='left')	
		option3 = OptionMenu(subframe5, vat, *ch_year)
		option3.pack(side='left',padx=10,pady=10)
		option3.configure(font="Verdana 10 bold",fg="black")
		bt7=Button(subframe5,text="OK",fg = "red",font="Verdana 8 bold",command=month_wise)
		bt7.pack(side='left',padx='10',pady=10)

		
	elif(dis=='Cardiotographic(CTG)'):
		attrs=['year','DP','DS.1','Tendency','CLASS']
		attr_vals=[['2012','2013','2014','2015','2016'],['0','1','2','3'],['0','0.001'],['1','0'],['1','2','3','4','5','6','7','8','9','10']]
		
		ch=['b','e','AC','FM','UC','DL','DS','DT','TR','LB','AC.1','FM.1','UC.1','DL.1','DS.1','DT.1','ASTV','MSTV','ALTV','MLTV','Width','Min','Max','Nmax','Nzeros','Mode','Mean','Median','Variance','Tendency','CLASS']
		
		for attr in attrs:		
				va = StringVar(root)
				va.set(attr)
				choices=attr_vals[count]
				option = OptionMenu(subframe, va, *choices)
				option.configure(font="Verdana 10 bold",fg="black")
				option.pack(padx='10',side='left',pady='5')
				attr_list.append(va)
				count=count+1
		
		bt1=Button(subframe,text="Submit",fg = "red",font="Verdana 8 bold",command=send)
		bt1.pack(anchor="center", pady='10')	

	#Geographica_Information_System
		
		w = Label(subframe2, text="njkengjs", bg="black", fg="black")
		w.pack(fill=X,anchor="center",pady='20')
		y = Label(subframe2, text="Geographical Information System",font="Verdana 15 bold", bg="orange", fg="black")
		y.pack(padx=5, pady=5, anchor='w')
		bt2=Button(subframe2,text="District-Wise Distribution",fg = "black",font="Verdana 10 bold",command=show_map1)
		bt2.pack(side='left',padx=10,pady=10)
		bt3=Button(subframe2,text="State-Wise Distribution",fg = "black",font="Verdana 10 bold",command=show_map2)
		bt3.pack(side='left',padx=10,pady=10)	
		bt4=Button(subframe2,text="Epicenter",fg = "black",font="Verdana 10 bold",command=epicenter)
		bt4.pack(side='left',padx=10,pady=10)	
		
	#Attribute_Wise_Distribution

		z = Label(subframe3, text="njkengjs", bg="black", fg="black")
		z.pack(fill=X,anchor="center",pady='20')
		a = Label(subframe3, text="Attribute based Distribution",font="Verdana 15 bold", bg="orange", fg="black")
		a.pack(padx=5, pady=5, anchor='w')
		option = OptionMenu(subframe3, vap, *ch)
		option.pack(side='left',padx=10,pady=15)
		option.configure(font="Verdana 10 bold",fg="black")
		bt5=Button(subframe3,text="Submit",fg = "red",font="Verdana 8 bold",command=col_dist)
		bt5.pack(side='left',padx='15',pady=15)	

		
	#Timeline

		c = Label(subframe4, text="njkengjs", bg="black", fg="black")
		c.pack(fill=X,anchor="center",pady='20')
		d = Label(subframe4, text="Timeline",font="Verdana 15 bold", bg="orange", fg="black")
		d.pack(padx=5, pady=5, anchor='w')
		
		#Location_affected_at_specific_time

		e = Label(subframe4, text=">> Location affected at specific Time",font="Verdana 12 bold", bg="orange", fg="black")
		e.pack(padx=5, pady=5, anchor='w')
		option1 = OptionMenu(subframe4, vas1, *ch_month)
		option1.pack(side='left',padx=10,pady=10)
		option1.configure(font="Verdana 10 bold",fg="black")
		option11 = OptionMenu(subframe4, vas11, *ch_year)
		option11.pack(side='left',padx=10,pady=10)
		option11.configure(font="Verdana 10 bold",fg="black")
		option22 = OptionMenu(subframe4, vas22, *ch_state)
		option22.pack(side='left',padx=10,pady=10)
		option22.configure(font="Verdana 10 bold",fg="black")
		option2 = OptionMenu(subframe4, vas2, *ch_form)
		option2.pack(side='left',padx=10,pady=10)
		option2.configure(font="Verdana 10 bold",fg="black")
		bt6=Button(subframe4,text="Submit",fg = "red",font="Verdana 8 bold",command=curr_areas)
		bt6.pack(side='left',padx='15',pady=10)
		
		#Time_wise_distribution
		
		f = Label(subframe5, text=">> Time-wise Distribution",font="Verdana 12 bold", bg="orange", fg="black")
		f.pack(padx=5, pady=5, anchor='w')	
		bt6=Button(subframe5,text="Year-wise Distribution",fg = "black",font="Verdana 10 bold",command=year_wise)
		bt6.pack(side='left',padx='20',pady=10)
		g = Label(subframe5, text="Month-wise Distribution ->",font="Verdana 10 bold", bg="orange", fg="Red")
		g.pack(padx=20, pady=10, side='left')	
		option3 = OptionMenu(subframe5, vat, *ch_year)
		option3.pack(side='left',padx=10,pady=10)
		option3.configure(font="Verdana 10 bold",fg="black")
		bt7=Button(subframe5,text="OK",fg = "red",font="Verdana 8 bold",command=month_wise)
		bt7.pack(side='left',padx='10',pady=10)
		
	elif(dis=='Cervical-Cancer'):
		attrs=['year','Smokes','Hormonal Contraceptives (years)','STDs:genital herpes','STDs:HIV','Dx:CIN','Dx:HPV',
				'Schiller']
		attr_vals=[['2012','2013','2014','2015','2016'],['0','1'],['0-2','2-4','4-6','6-8','8-10','10-12','12-14','14-16','16-18','18-20'],['0','1'],['0','1'],['0','1'],['0','1'],['0','1']]
		
		ch=['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)','STDs','STDs (number)','STDs:condylomatosis','STDs:cervical condylomatosis',
		'STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV','STDs: Number of diagnosis',
		'STDs: Time since first diagnosis','STDs: Time since last diagnosis','Dx:Cancer','Dx:CIN','Dx:HPV','Dx,Hinselmann','Schiller','Citology']
		
		for attr in attrs:		
				va = StringVar(root)
				va.set(attr)
				choices=attr_vals[count]
				option = OptionMenu(subframe, va, *choices)
				option.configure(font="Verdana 10 bold",fg="black")
				option.pack(padx='10',side='left',pady='5')
				attr_list.append(va)
				count=count+1	
		
		bt1=Button(subframe,text="Submit",fg = "red",font="Verdana 8 bold",command=send)
		bt1.pack(anchor="center", pady='10')	

	#Geographica_Information_System
		
		w = Label(subframe2, text="njkengjs", bg="black", fg="black")
		w.pack(fill=X,anchor="center",pady='20')
		y = Label(subframe2, text="Geographical Information System",font="Verdana 15 bold", bg="orange", fg="black")
		y.pack(padx=5, pady=5, anchor='w')
		bt2=Button(subframe2,text="District-Wise Distribution",fg = "black",font="Verdana 10 bold",command=show_map1)
		bt2.pack(side='left',padx=10,pady=10)
		bt3=Button(subframe2,text="State-Wise Distribution",fg = "black",font="Verdana 10 bold",command=show_map2)
		bt3.pack(side='left',padx=10,pady=10)	
		bt4=Button(subframe2,text="Epicenter",fg = "black",font="Verdana 10 bold",command=epicenter)
		bt4.pack(side='left',padx=10,pady=10)	
		
	#Attribute_Wise_Distribution

		z = Label(subframe3, text="njkengjs", bg="black", fg="black")
		z.pack(fill=X,anchor="center",pady='20')
		a = Label(subframe3, text="Attribute based Distribution",font="Verdana 15 bold", bg="orange", fg="black")
		a.pack(padx=5, pady=5, anchor='w')
		option = OptionMenu(subframe3, vap, *ch)
		option.pack(side='left',padx=10,pady=15)
		option.configure(font="Verdana 10 bold",fg="black")
		bt5=Button(subframe3,text="Submit",fg = "red",font="Verdana 8 bold",command=col_dist)
		bt5.pack(side='left',padx='15',pady=15)	

		
	#Timeline

		c = Label(subframe4, text="njkengjs", bg="black", fg="black")
		c.pack(fill=X,anchor="center",pady='20')
		d = Label(subframe4, text="Timeline",font="Verdana 15 bold", bg="orange", fg="black")
		d.pack(padx=5, pady=5, anchor='w')
		
		#Location_affected_at_specific_time

		e = Label(subframe4, text=">> Location affected at specific Time",font="Verdana 12 bold", bg="orange", fg="black")
		e.pack(padx=5, pady=5, anchor='w')
		option1 = OptionMenu(subframe4, vas1, *ch_month)
		option1.pack(side='left',padx=10,pady=10)
		option1.configure(font="Verdana 10 bold",fg="black")
		option11 = OptionMenu(subframe4, vas11, *ch_year)
		option11.pack(side='left',padx=10,pady=10)
		option11.configure(font="Verdana 10 bold",fg="black")
		option22 = OptionMenu(subframe4, vas22, *ch_state)
		option22.pack(side='left',padx=10,pady=10)
		option22.configure(font="Verdana 10 bold",fg="black")
		option2 = OptionMenu(subframe4, vas2, *ch_form)
		option2.pack(side='left',padx=10,pady=10)
		option2.configure(font="Verdana 10 bold",fg="black")
		bt6=Button(subframe4,text="Submit",fg = "red",font="Verdana 8 bold",command=curr_areas)
		bt6.pack(side='left',padx='15',pady=10)
		
		#Time_wise_distribution
		
		f = Label(subframe5, text=">> Time-wise Distribution",font="Verdana 12 bold", bg="orange", fg="black")
		f.pack(padx=5, pady=5, anchor='w')	
		bt6=Button(subframe5,text="Year-wise Distribution",fg = "black",font="Verdana 10 bold",command=year_wise)
		bt6.pack(side='left',padx='20',pady=10)
		g = Label(subframe5, text="Month-wise Distribution ->",font="Verdana 10 bold", bg="orange", fg="Red")
		g.pack(padx=20, pady=10, side='left')	
		option3 = OptionMenu(subframe5, vat, *ch_year)
		option3.pack(side='left',padx=10,pady=10)
		option3.configure(font="Verdana 10 bold",fg="black")
		bt7=Button(subframe5,text="OK",fg = "red",font="Verdana 8 bold",command=month_wise)
		bt7.pack(side='left',padx='10',pady=10)

	
	
	
	
root = Tk()
root.title("AROGYA")
root.configure(background="black")

menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New", command=restart_program)
filemenu.add_command(label="Open", command=donothing)
filemenu.add_command(label="Save", command=donothing)
filemenu.add_command(label="Save as...", command=donothing)
filemenu.add_command(label="Close", command=donothing)

filemenu.add_separator()

filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Undo", command=donothing)

editmenu.add_separator()

editmenu.add_command(label="Cut", command=donothing)
editmenu.add_command(label="Copy", command=donothing)
editmenu.add_command(label="Paste", command=donothing)
editmenu.add_command(label="Delete", command=donothing)
editmenu.add_command(label="Select All", command=donothing)

menubar.add_cascade(label="Edit", menu=editmenu)
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help Index", command=donothing)
helpmenu.add_command(label="About...", command=about)
menubar.add_cascade(label="Help", menu=helpmenu)

root.config(menu=menubar)

topframe=Frame(root)
topframe.pack(fill=X)
middleframe=Frame(root,bg="white")
middleframe.pack(side=LEFT,fill=BOTH,expand=True)


canvas=Canvas(middleframe)
canvas.pack(side=RIGHT,fill=BOTH,expand=True)
frame=Frame(canvas)

can_frame=canvas.create_window((0,0),window=frame,anchor='nw')

myscrollbar=Scrollbar(canvas,orient="vertical",command=canvas.yview)
canvas.configure(yscrollcommand=myscrollbar.set)
myscrollbar.pack(side="right",fill="y")

frame.bind("<Configure>",myfunction)
canvas.bind("<Configure>",myfunction2)


subframe0=Frame(frame,bg="white")
subframe=Frame(frame,bg="orange")
subframe2=Frame(frame,bg="orange")
subframe3=Frame(frame,bg="orange")
subframe4=Frame(frame,bg="orange")
subframe5=Frame(frame,bg="orange")
subframe6=Frame(frame,bg="orange")

subframe0.pack(fill=BOTH,anchor=CENTER)
subframe.pack(fill=BOTH,anchor=CENTER)
subframe2.pack(fill=BOTH,anchor=CENTER)
subframe3.pack(fill=BOTH,anchor=CENTER)
subframe4.pack(fill=BOTH,anchor=CENTER)
subframe5.pack(fill=BOTH,anchor=CENTER)
subframe6.pack(fill=BOTH,anchor=CENTER)


welcome_label=Label(topframe, text="AROGYA",
		 fg = "white",
		 bg = "black",
		 font = "Verdana 40 bold")

welcome_label.pack(fill=X)

var = StringVar(root)
var.set('Choose your Disease')
vap = StringVar(root)
vap.set("Choose your Attribute")
vas1 = StringVar(root)
vas1.set("Choose month")
vas11= StringVar(root)
vas11.set("Choose year")
vas22= StringVar(root)
vas22.set("Choose State")
vas2 = StringVar(root)
vas2.set("Choose list or map")
vat = StringVar(root)
vat.set("Choose year")
choices = ['Heart-Disease','Cervical-Cancer','Cardiotographic(CTG)','Chronic Kidney Disease']
ch_month = [1,3,4,5,6,7,8,9,10,11,12]
ch_form = ['List','Map']
ch_state = ['All','Madhya Pradesh','Rajasthan','Uttar Pradesh']
ch_year = [2012,2013,2014,2015,2016]
optionk = OptionMenu(subframe0, var, *choices)
optionk.pack(anchor='center',pady='10')
optionk.configure(font="Verdana 12",fg="black")

btn=Button(subframe0,text="Submit",fg = "red",font="Verdana 8 bold",command=select)
btn.pack(anchor='center',pady='5')


root.mainloop()
