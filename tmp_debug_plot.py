import numpy as np
from types import SimpleNamespace
from utils.plotting import plot_model_comparison, plot_model_comparison_3d

x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])
res = SimpleNamespace(name='test', model_type='linear', params={'a': 1, 'b': 0, 'c': 0}, equation='y=x', r2=1.0, rmse=0.0)

html2 = plot_model_comparison(x, y, [res], 'N', 'phi', point_labels=np.array(['A', 'B', 'C', 'D']))
html3 = plot_model_comparison_3d(x, np.array([1, 2, 3, 4]), y, res, 'N', 'F', 'phi', point_labels=np.array(['A', 'B', 'C', 'D']))

print('2d markers', 'markers' in html2 and 'Scatter' in html2)
print('3d scatter3d', 'scatter3d' in html3.lower())
print('2d length', len(html2))
print('3d length', len(html3))
print('2d snippet', html2[:400])
print('3d snippet', html3[:400])
