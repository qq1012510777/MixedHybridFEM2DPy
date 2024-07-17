import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, coo_matrix

row_indices_A = []
col_indices_A = []
values_A = []

def add_triplets_A(triplets):
    for (i, j, value) in triplets:
        row_indices_A.append(i)
        col_indices_A.append(j)
        values_A.append(value)

row_indices_b = []
col_indices_b = []
values_b = []

def add_triplets_b(triplets):
    for (i, j, value) in triplets:
        row_indices_b.append(i)
        col_indices_b.append(j)
        values_b.append(value)

# Define the vertices of the polygon (a simple square for this example)
vertices = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

# Create a grid of points within the polygon
x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()

# Create a Delaunay triangulation of the points
triangulation = tri.Triangulation(x, y)

# Extract points, triangles, and edges
P = np.vstack((x, y)).T  # Points
T = triangulation.triangles  # Triangles (indices of points)
# E = triangulation.edges  # Edges (pairs of indices of points)

#----------------------------------------------------------------------------------------------------
NumNodes = np.shape(P)[0]
NumElements = np.shape(T)[0]

temp1 = coo_matrix((np.ones((NumElements * 3)) * NumElements * 3, (np.concatenate((T[:, 0], T[:, 1], T[:, 2]), axis=0), np.concatenate((T[:, 1], T[:, 2], T[:, 0]), axis=0))), shape=(NumNodes, NumNodes))
PntIDToEdgeID = temp1.tolil()

InternalEdgeID = 1 # from 1
for i in range(0, NumElements):
    for j in range(0, 3):
        if  PntIDToEdgeID[T[i, (j + 1) % 3], T[i, j]] != 0:
            if PntIDToEdgeID[T[i, (j + 1) % 3], T[i, j]] == NumElements * 3:
                PntIDToEdgeID[T[i, j], T[i, (j + 1) % 3]] = InternalEdgeID
                InternalEdgeID = InternalEdgeID + 1
                #print(EdgeID, i, j, T[i, j], T[i, (j + 1) % 3], "new internal edge", PntIDToEdgeID[T[i, j], T[i, (j + 1) % 3]])
            else:
                PntIDToEdgeID[T[i, j], T[i, (j + 1) % 3]] = PntIDToEdgeID[T[i, (j + 1) % 3], T[i, j]] 

M = np.array([
    [2., 0., 1., 0., 1., 0.],
    [0., 2., 0., 1., 0., 1.],
    [1., 0., 2., 0., 1., 0.],
    [0., 1., 0., 2., 0., 1.],
    [1., 0., 1., 0., 2., 0.],
    [0., 1., 0., 1., 0., 2.]])

NonFluxListEdge = []
for i in range(0, NumElements):
    P1 = P[T[i, 0], :]
    P2 = P[T[i, 1], :]
    P3 = P[T[i, 2], :]

    T_area = np.zeros((3,3))
    T_area[0:2, 0] = P1.transpose()
    T_area[0:2, 1] = P2.transpose()
    T_area[0:2, 2] = P3.transpose()
    T_area[2,:] = [1, 1, 1]
    T_area = 0.5 * np.linalg.det(T_area)

    N = np.zeros((6, 3))
    N[2:4, 0] = (P2 - P1).transpose()
    N[4:6, 0] = (P3 - P1).transpose()

    N[0:2, 1] = (P1 - P2).transpose()
    N[4:6, 1] = (P3 - P2).transpose()

    N[0:2, 2] = (P1 - P3).transpose()
    N[2:4, 2] = (P2 - P3).transpose()
    
    C = np.array([
        [np.linalg.norm(P3-P2), 0, 0],
        [0, np.linalg.norm(P3-P1), 0],
        [0, 0, np.linalg.norm(P1-P2)]
    ])

    B = 1. / 48 / T_area * C.transpose() @ N.transpose() @ M @ N @ C

    for j in range(0, 3):
        P_local_1_ID = T[i, (j + 1) % 3]
        P_local_2_ID = T[i, (j + 2) % 3]
        Length_pp = np.linalg.norm(P[P_local_1_ID, :] - P[P_local_2_ID, :])
        if PntIDToEdgeID[P_local_1_ID, P_local_2_ID] != NumElements * 3:
            InternalEdgeID_global = PntIDToEdgeID[P_local_1_ID, P_local_2_ID] - 1

            triplets4 = [(InternalEdgeID_global + NumElements * 4, i * 3 + j, -Length_pp), 
                         (i * 3 + j, InternalEdgeID_global + NumElements * 4, -Length_pp)]   
            add_triplets_A(triplets4)
            # print(P_local_1_ID, P_local_2_ID, "internal", i * 3 + j, InternalEdgeID_global + NumElements * 4, InternalEdgeID_global)
        # boundary condition ----------------------Dirchilet
        if P[P_local_1_ID, 0] == 0. and P[P_local_2_ID, 0] == 0.:
            triplets5 = [(i * 3 + j, 0, 100 * Length_pp)]
            add_triplets_b(triplets5)
        if P[P_local_1_ID, 0] == 1. and P[P_local_2_ID, 0] == 1.:
            triplets6 = [(i * 3 + j, 0, 1 * Length_pp)]
            add_triplets_b(triplets6)
        if (P[P_local_1_ID, 1] == 0. and P[P_local_2_ID, 1] == 0.) or (P[P_local_1_ID, 1] == 1. and P[P_local_2_ID, 1] == 1.): # neumann
            B[j, :] = 0
            B[:, j] = 0
            B[j, j] = 1
            C[j, j] = 0  
            #print(P_local_1_ID, P_local_2_ID, "non flux", B, C)

    triplets1 = [(i * 3 + 0, i * 3 + 0, B[0, 0]), (i * 3 + 0, i * 3 + 1, B[0, 1]), (i * 3 + 0, i * 3 + 2, B[0, 2]),
                 (i * 3 + 1, i * 3 + 0, B[1, 0]), (i * 3 + 1, i * 3 + 1, B[1, 1]), (i * 3 + 1, i * 3 + 2, B[1, 2]),
                 (i * 3 + 2, i * 3 + 0, B[2, 0]), (i * 3 + 2, i * 3 + 1, B[2, 1]), (i * 3 + 2, i * 3 + 2, B[2, 2]),]
    add_triplets_A(triplets1)

    triplets2 = [(i * 3 + 0, i + NumElements * 3, C[0, 0]), 
                 (i * 3 + 1, i + NumElements * 3, C[1, 1]), 
                 (i * 3 + 2, i + NumElements * 3, C[2, 2])]
    add_triplets_A(triplets2)

    triplets3 = [(i + NumElements * 3, i * 3 + 0, C[0, 0]), 
                 (i + NumElements * 3, i * 3 + 1, C[1, 1]), 
                 (i + NumElements * 3, i * 3 + 2, C[2, 2])]
    add_triplets_A(triplets3)

    #-----------------------------------
    # B_c = np.zeros((3, 3))
    # for j in range(0, 3):
    #     for k in range(0, 3):
    #         E_j =  np.linalg.norm(P[T[i, (j + 1) % 3], :] - P[T[i, (j + 2) % 3], :])
    #         E_k =  np.linalg.norm(P[T[i, (k + 1) % 3], :] - P[T[i, (k + 2) % 3], :])
    #         for l in range(0, 3):
    #             for m in range(0, 3):
    #                 delta_lm = 0
    #                 if l == m:
    #                     delta_lm = 1
    #                 B_c[j, k] = B_c[j, k] + T_area / 12 * (1 + delta_lm) * np.dot((P[T[i, l], :] -  P[T[i, j], :]), (P[T[i, m], :] -  P[T[i, k], :]))
    #         B_c[j, k] = B_c[j, k] * E_j * E_k / (4 * T_area ** 2)

# print(EdgeID, InternalEdgeID_loop)
# print(row_indices_A)
# print(col_indices_A)
NumInternalEdge = InternalEdgeID - 1
A = coo_matrix((values_A, (row_indices_A, col_indices_A)), shape=(NumElements * 3 + NumElements + NumInternalEdge, NumElements * 3 + NumElements + NumInternalEdge))

# Convert to CSR format for efficient arithmetic and matrix vector operations
A = A.tocsr()

b = coo_matrix((values_b, (row_indices_b, col_indices_b)), shape=(NumElements * 3 + NumElements + NumInternalEdge, 1))
b = b.tocsr()

# for row in b.toarray():
#     formatted_row = ' '.join(f'{elem:{1}.2f}' for elem in row)
#     print(formatted_row)

x = sp.linalg.spsolve(A, b)
PressureElement = x[NumElements * 3: NumElements * 4 ]

#----------------------------------------------------------------------------------------------------
# Plot the polygon and the mesh
plt.figure(figsize=(8, 8))
plt.gca().set_aspect('equal')

# Plot the polygon
polygon = plt.Polygon(vertices, edgecolor='k', fill=None)
plt.gca().add_patch(polygon)

# Plot the triangulation
plt.triplot(triangulation, '-', markersize=5)
plt.tripcolor(triangulation, PressureElement, cmap='viridis', edgecolors='k', linewidth=0.5)
#for i in range(0, NumNodes):
#    plt.text(P[i, 0], P[i, 1], str(i))
plt.colorbar()
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.title("2D Polygon Mesh")
plt.show()
