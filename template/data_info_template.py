# template file for data visualizing

import numpy as np 
import matplotlib.pyplot as plt 

from statslib.main.pca import PrincipalComponentAnalysis
from MLEK.tools.data_tools import DataTools, euclidean_distance
from MLEK.tools.plot_tools import *

file_name = '../example_demo/demo_data'
data = DataTools(file_name, PrincipalComponentAnalysis, euclidean_distance)
data.load_data(n_lines=100, start_column=2)

pairwise_distance = data.pairwise_distance()
plot_distance(pairwise_distance, bins=50)
principal_components = data.fit_transform(n_components=10)
plot_principal_components(principal_components, hist_params={'bins': 50})

dens_x = data.raw_data_
plot_real_space_density(dens_x)
