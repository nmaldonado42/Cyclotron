#################### Numerical Simulation of a Cyclotron ####################
### Nicolas Maldonado Baracaldo
### 201423809
#############################################################################

import numpy as np
import matplotlib.pyplot as plt

### Cyclotron parameters: ###
E = np.array([6000000,0,0]) #Electric field
B = np.array([0,0,-1.64])   #Magnetic field
R = 0.923                   #Radius
d = 0.120                   #Distance between dees

### Particle parameters: ###
q = 1.6021766*10**-19 #Particle charge
m = 1.6726219*10**-27 #Particle mass

### Simulation parameters: ###
r_0 = np.array([0,0.05,0]) #Initial position
v_0 = np.array([100,0,0])  #Initial velocity
t_max = 1.3*10**-6         #Simulation running time
N = 10000                  #Number of samples for simulation

### Function to calculate Lorentz force: ###
def F_Lorentz(r, v, t):
    return q*E*np.cos((q*B[2]/m)*t)*np.heaviside(r[0]+d/2, 1)*np.heaviside(-r[0]+d/2, 1)\
            + q*np.cross(v, B)
#end of F_Lorentz() function

### Function to update r, v, a via 4th order Runge-Kutta: ###
def RK4(t):
    global r
    global v
    global a
    global dt
    
    #First step:
    k1 = v
    l1 = a
    
    #Second step:
    k2 = v + dt/2*l1
    l2 = F_Lorentz(r+dt/2*k1, k2, t+dt/2)/m
    
    #Third step:
    k3 = v + dt/2*l2
    l3 = F_Lorentz(r+dt/2*k2, k3, t+dt/2)/m
    
    #Fourth step:
    k4 = v + dt*l3
    l4 = F_Lorentz(r+dt*k3, k4, t+dt)/m
    
    #Averages:
    k = (k1 + 2*k2 + 2*k3 + k4)/6
    l = (l1 + 2*l2 + 2*l3 + l4)/6

    #Update acceleration:
    a = F_Lorentz(r, v, t+dt)/m
    #Update velocity
    v = v + dt*l
    #Update position
    r = r + dt*k
#end of RK4() function

t = np.linspace(0,t_max,N) #Times array
dt = t[1] - t[0]           #Time step
r = r_0                    #Set initial position
v = v_0                    #Set initial velocity
a = F_Lorentz(r, v, 0)/m   #Set initial acceleration

x = np.array([r[0]]) #X positions array
y = np.array([r[1]]) #Y positions array

p_x = np.array([m*v[0]/(5.36*10**-22)]) #X momenta array
p_y = np.array([m*v[1]/(5.36*10**-22)]) #Y momenta array

r_xy = np.array([np.sqrt(r[0]**2 + r[1]**2)])                  #Position-based radii array
r_p = np.array([np.sqrt((m*v[0])**2 + (m*v[1])**2)/(-q*B[2])]) #Momentum-based radii array

### Loop over times ###
for i in range(1,t.size):
    #Update variables:
    RK4(t[i-1])
    
    #Store positions:
    x = np.append(x, [r[0]])
    y = np.append(y, [r[1]])
    
    #Store momenta:
    p_x = np.append(p_x, [m*v[0]/(5.36*10**-22)])
    p_y = np.append(p_y, [m*v[1]/(5.36*10**-22)])
    
    #Store radii:
    r_xy = np.append(r_xy, [np.sqrt(r[0]**2 + r[1]**2)])
    r_p = np.append(r_p, [np.sqrt((m*v[0])**2 + (m*v[1])**2)/(-q*B[2])])
    
    #Stop loop when approaching radius:
    if r[0] > 0.935*R:
        break
    #end if statement
#end of for loop

#Fix null acceleration:
a = np.array([0,0,0])

### Loop over remaining times ###   
for i in range(x.size,t.size):
    #Update position:
    r = r + dt*v
    
    #Store positions:
    x = np.append(x, [r[0]])
    y = np.append(y, [r[1]])
    
    #Store momenta:
    p_x = np.append(p_x, [m*v[0]/(5.36*10**-22)])
    p_y = np.append(p_y, [m*v[1]/(5.36*10**-22)])
    
    #Store radii:
    r_xy = np.append(r_xy, [np.sqrt(r[0]**2 + r[1]**2)])
    r_p = np.append(r_p, [np.sqrt((m*v[0])**2 + (m*v[1])**2)/(-q*B[2])])
    
    #Stop loop once radius is reached:
    if r[1] > R:
        break
    #end if statement
#end of for loop
   
p_l = np.linspace(0.5211189*np.pi,1.4788811*np.pi,100)   #Angles for left dee
x_l = np.append([-d/2,-d/2],0.906996*np.cos(p_l))        #X coordinates of left dee
y_l = np.append([-0.86,0.95],0.906996*np.sin(p_l)+0.045) #Y coordinates of left dee

p_r = np.linspace(-0.4788811*np.pi,0.4788811*np.pi,100)  #Angles for right dee
x_r = np.append([d/2,d/2],0.906996*np.cos(p_r))          #X coordinates of right dee
y_r = np.append([0.95,-0.86],0.906996*np.sin(p_r)+0.045) #Y coordinates of right dee

#Plot y positions vs. x positions:
plt.figure(figsize=[12,12])
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.xlabel(r'$x$ (m)',fontsize=18)
plt.ylabel(r'$y$ (m)',fontsize=18)
plt.plot(x,y,color='#74AAE1',linewidth=2)
plt.plot(x_l,y_l,color='grey')
plt.plot(x_r,y_r,color='grey')
plt.savefig('path.png')
plt.close()

#Plot x positions vs. x momenta:
plt.figure(figsize=[12,12])
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.xlabel(r'$p_x$ (MeV/c)',fontsize=20)
plt.ylabel(r'$x$ (m)',fontsize=20)
plt.plot(p_x,x,color='#74AAE1',linewidth=2)
plt.savefig('momentum_x.png')
plt.close()

#Plot y positions vs. y momenta:
plt.figure(figsize=[12,12])
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.xlabel(r'$p_y$ (MeV/c)',fontsize=20)
plt.ylabel(r'$y$ (m)',fontsize=20)
plt.plot(p_y,y,color='#74AAE1',linewidth=2)
plt.savefig('momentum_y.png')
plt.close()

#Plot position-based radii vs. momentum-based radii:
plt.figure(figsize=[12,12])
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.xlabel(r'$r_p$ (m)',fontsize=20)
plt.ylabel(r'$r_{xy}$ (m)',fontsize=20)
plt.plot(r_p,r_xy,color='#74AAE1',linewidth=2)
plt.savefig('radii.png')
plt.close()