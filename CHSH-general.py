import numpy as np
from numpy import linalg as LA
import sys


from scipy import optimize


import csv

############### Some not important stuff  #################

DEBUG_ENABLED = False
Verbose = False

import os
# System call
os.system("")

# Class of different styles
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


############### END not important stuff here #################


#given a column vector (ket/sum of kets) returns the density matrix/operator.
def cvrt2density(psi):
   if DEBUG_ENABLED: print ("[cvrt2density]got:",psi)
   psi_H = psi.conj().T
   if DEBUG_ENABLED: print ("[cvrt2density]the conj Tr:",psi_H)
   return np.matmul(psi,psi_H)


# given a measurement basis, returns the measurement projectors (as an array)
def Meas_observable(M_obs):
   w, v = LA.eig(M_obs)
   
   nb_eigenv = len(w)
   ret = dict()
   for i in range(nb_eigenv):
      lambda_val = np.ceil(w[i]) if w[i]>0 else np.floor(w[i])
      if DEBUG_ENABLED: print(lambda_val, v[:,i])
      vec = (v[:,i])
      vec = vec.reshape(vec.size,1)
      
      projector = cvrt2density(vec)
      if DEBUG_ENABLED: print("proj:",projector)
      
      ret.update({lambda_val : projector })
   return ret

# given a density matrix and two projectors, it returns the pr. of using trace [(P1.P2)rho]
def prob(rho, P1, P2):
   M = np.matmul(rho, np.kron( P1, P2 )   )
   pr = np.trace(  M  )
   return pr
   
# given a quantum state (np matrix), it applies some noise and alters the state
def noise_channel(state_density, eps):

   
   Id4Div4 =  np.array([[1, 0,0,0],
                        [0, 1,0,0],
                        [0,0, 1,0],
                        [0,0, 0,1] ])*(1/4)
   return  (state_density*(1-eps)) + (( Id4Div4*eps)*np.trace(state_density))

def CHSH (angle,Epsillon, a0, a1, b0,b1):
   init_state =  np.array([[ np.cos ( angle)],
                              [0],
                              [0],
                              [np.sin(angle)] ])

   init_density = cvrt2density(init_state)
   noisy_density = noise_channel(init_density,Epsillon)
   
   

   pauliZ =  np.array([ [1, 0],
                        [0, -1] ])
   
   pauliX =  np.array([ [0, 1],
                        [1, 0] ])   
   
   Id =  np.array([ [1, 0],
                     [0, 1] ])
   Id2 =  Id *(1/2)

   A00 = Id2 + (( pauliZ * np.cos(a0) +  pauliX*np.sin(a0) ) *1/2)
   A10 = Id - A00
   
   A01 = Id2 + (( pauliZ * np.cos(a1) +  pauliX*np.sin(a1) ) *1/2)
   A11 = Id - A01
   
   B00 = Id2 + (( pauliZ * np.cos(b0) +  pauliX*np.sin(b0) ) *1/2)
   B10 = Id - B00

   B01 = Id2 + (( pauliZ * np.cos(b1) +  pauliX*np.sin(b1) ) *1/2)
   B11 = Id - B01
   
   #there are 4 cases to win, 2 options for each case, so in total 8 probabilites to sum:

   #P(Win| X = 0, Y = 0) =  P(A=0, B=0| X = 0, Y = 0) + P(A=1, B=1| X = 0, Y = 0)
   pwin = prob(noisy_density, A00, B00) + prob (noisy_density, A10,B10)

   #P(Win| X = 0, Y = 1) =  P(A=0, B=0| X = 0, Y = 1) + P(A=1, B=1| X = 0, Y = 1)
   pwin += prob(noisy_density, A00, B01) + prob (noisy_density, A10,B11)

   #P(Win| X = 1, Y = 0) =  P(A=0, B=0| X = 1, Y = 0) + P(A=1, B=1| X = 1, Y = 0)
   pwin += prob(noisy_density, A01, B00) + prob (noisy_density, A11,B10)

   #P(Win| X = 1, Y = 1) =  P(A=0, B=1| X = 1, Y = 1) + P(A=1, B=0| X = 1, Y = 1)
   pwin += prob(noisy_density, A01, B11) + prob (noisy_density, A11,B01)
   
   return (pwin/4)

# a function that returns the opposite of the CHSH probability since the optimization module we use can only minimize
# so we just do things in a reversed way



def CHSHoptimize (  angle = np.pi/4, Epsillon=0):

   def CHSH_neg (x):
      a0, a1, b0,b1 = x
      return - CHSH(angle,Epsillon, a0, a1, b0,b1)

   x0 =[0,0,0,0]
   b = (-np.pi/4, np.pi/4)
   bnds = (b,b,b,b)
   #res = optimize.minimize(CHSH_neg, x0, method='SLSQP', bounds=bnds)
   res = optimize.minimize(CHSH_neg, x0, method='Powell', bounds=bnds)
   if Verbose:
      print( style.YELLOW +f"we get :{-res.fun:} for {res.x}")
      print( style.RESET +f"with this optimization we get up to 0.82, not bad, but we expect to reach 0.85")
      print("This is probably due to FP so some rounding errors may occur. if we remove the bounds:")
   
   res = optimize.minimize(CHSH_neg, x0, method='Powell')
   return res

if __name__ == "__main__":
   print("welcome to CHSH calculator")
   print("This program shows that the winning probablity with an optimal strategy can reach 0.85")
   
   # at first we try with best known values just to check if our program is correct and without noise
   best_params = [0, np.pi/2, np.pi/4 ,-np.pi/4]
   best_known = CHSH (np.pi/4, 0, best_params[0],best_params[1],best_params[2],best_params[3])
   if Verbose:
      print( style.GREEN + f"\nwinning prob with best known parameters {best_params} is: {best_known}")
      print( style.RESET +f"\nAssuming we did not know the best angles, given the angle and epsillon can we use an optimization method to find them? let's try!\n")

   angle = np.pi/4
   Epsillon=0
   res = CHSHoptimize()
   
   if Verbose:
      print( style.GREEN +  f"we get :{-res.fun:} for {res.x}")
      print(f"Now we get the 0.85 that we expect!!")
      print( style.RESET)


   if len(sys.argv)>2 : print ("Oups, I only take the filename as argument, no more parameters :(")
   if len(sys.argv)>1 : 
      filepath = str(sys.argv[1])

      outf = open("[M]theta_noise_Pscore,Mscore.csv", "a")
      

      with open(filepath, 'r') as file:
         reader = csv.reader(file)
         for row in reader:
            angle = float(row[0])
            eps =float(row[1])
            Pscore =float(row[2])
            res = (CHSHoptimize(angle,eps))
            print(f"{angle},{eps},{-res.fun}\n")
            outf.write(f"{angle},{eps},{Pscore},{-res.fun}\n")
      
      outf.close()
   
   
   
   
   
   
   
   #             options={'xtol': 1e-8, 'disp': True})
   # res = optimize.minimize(CHSH_neg, x0, args=(2,), method='nelder-mead',
   #             options={'xtol': 1e-8, 'disp': True})
