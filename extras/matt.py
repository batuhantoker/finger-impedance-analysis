"""FEA results visualization for prosthesis design.

Reads simulation results (thickness, rotation, reaction force, contact area)
from an Excel file and creates 3D scatter plots with color-coded metrics.
Unrelated to the EMG pipeline.
"""

import pandas as pd
from finger_impedance.core.functions import *

df = pd.read_excel('FINAL_Results.xlsx')
df = df.dropna()
df = df.drop(['Name'], axis=1)
df = df.sort_values(by=['Thickness', 'Rotation'])
print(df.head())

x = df['Thickness']
y1 = df['Max Reaction Force']
y2 = df['Contact m^2']

plt.plot(x, y1)
plt.plot(x, y2)

ax3 = df.plot('Rotation', y=['Max Reaction Force'], kind="bar", title=" ")
ax3.set_ylabel("", fontsize=15)
ax3.set_xlabel("Thickness", fontsize=15)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
im = ax.scatter(df['Thickness'], df['Exact_rotation'], df['Count'], c=df['Max Reaction Force'])
ax.set_xlabel('Thickness')
ax.set_ylabel('Rotation')
ax.set_zlabel('Count')
fig.colorbar(im, ax=ax, location='left', label='Max reaction force N')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
im2 = ax2.scatter(df['Thickness'], df['Exact_rotation'], df['Count'], c=df['Contact m^2'])
ax2.set_xlabel('Thickness')
ax2.set_ylabel('Rotation')
ax2.set_zlabel('Count')
fig2.colorbar(im2, ax=ax2, location='left', label='Contact m2')
plt.show()
