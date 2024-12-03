'''
Run to remove output from Jupyter notebook before committing
'''

import os

os.system('jupyter nbconvert --clear-output workbook.ipynb --inplace')