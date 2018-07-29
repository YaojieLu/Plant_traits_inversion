import xlrd
import numpy as np
import theano
import theano.tensor as tt
import os

# Read xlsx file
os.chdir("..")
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
species = 'Pm_RN_S'#'Pm_RN_S' 'Am_RN_S' 'Nd_RN_S' 'Qc_RN_S' 'Qg_SN_S'

# Get data from column with specified colname
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:61])
Rf = get_data('Rf')
vn = get_data(species)

sE = tt.dvector('spE')
sR = tt.dvector('sR')
ss = tt.dscalar('ss')

output, updates = theano.scan(fn = lambda E, R, s : tt.minimum(s - E + R, 1),
                              sequences = [sE, sR],
                              outputs_info = [ss])

f = theano.function(inputs = [sE, sR, ss],
                    outputs = output,
                    updates = updates)

print(f(vn * 0.013, Rf/1000/2/0.43*0.7, 0.6))