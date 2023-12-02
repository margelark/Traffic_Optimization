import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

initial_matrix = "final project math 214.xlsx"
adjusted_matrix = "adjusted_matrix.xlsx"

df = pd.read_excel(adjusted_matrix)
matrix = df.to_numpy()


#A = np.array(
#    [[.1,.4,.4,0],
#     [.5,.1,0,.5],
#     [.4,0,.1,.4],
#     [0,.5,.5,.1]])

#initial state vector
Initial_state = np.array(
    [[0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826],
     [0.04347826]])

T = np.linalg.matrix_power(matrix, 20)
result = np.dot(T, Initial_state)
new_result = result
print(new_result)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

index_eigenvalue_1 = np.where(np.isclose(eigenvalues, 1))[0]

# Print the corresponding eigenvector(s)
if len(index_eigenvalue_1) > 0:
    eigenvector_1 = eigenvectors[:, index_eigenvalue_1]
    print("Eigenvector(s) corresponding to eigenvalue 1:")
    num_cars = eigenvector_1 * -1000
    print(num_cars)
else:
    print("No eigenvector corresponding to eigenvalue 1.")

 # array to hold each future state vector
xs = np.zeros((23,23))
x = np.array(
    [0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826, 0.04347826])

# compute future state vectors
for i in range(23):
    xs[i] = x
    #print(f'x({i}) = {x}')
    x = matrix @ x

# Plot the evolution of each element over time with dots
iterations = np.arange(1, 24)

# Plot for each element from x1 to x23
for j in range(23):
    plt.plot(iterations, xs[:, j], marker='o', label=f'x{j + 1}')

# Add labels and a legend
plt.xlabel('Iteration')
plt.ylabel('Value')
#plt.legend()

# Show the plot
plt.show()

