#Please see the attached SAE publication.  
# It outlines how to model a non-linear battery model.
#  The required parameters for the battery include a
#           nominal voltage of 96V,
#           an amp-hour capacity of 420 Ahr, and 
#           the chemical composition of Li-Ion.  
# Regarding the fuel consumption modeling, 
# please see the attached Excel data provided to me by TARDEC (now GVSC). 
# 
# Regarding the fuel consumption of the diesel generator, we will want to us a 
# polynomial fit such that at zero load there is no fuel consumption, using the 5kW variant. 
# 
# Let me know if you have any questions, or if there is any additional way I can assist. Thanks

import numpy as np

vInitial = 
K = 
q = 
q_it = 
A = vFull
B_it = 


vBatt = vInitial - K * (q / q_it) + A * np.e^(-B_it)