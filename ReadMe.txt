Two scripts are working together "Main" and "functions". 
The functions script has all the functions needed/used in main with the proper explanations and descriptions for each one. 

To run the code from notebook (functions3 and main3.ipynb) use:

%run [address where function is located]/functions3.ipynb

If it is needed to run it from Spyder (mainV5 and functions.py) use:

import functions
import subprocess
subprocess.call(['python', 'functions.py'])