'''
Run to remove output from Jupyter notebook before committing
'''

import os

os.system('jupyter nbconvert --clear-output andy_workbook.ipynb --inplace')
os.system('jupyter nbconvert --clear-output ising_workbook.ipynb --inplace')